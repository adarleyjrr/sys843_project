"""Fast regression checks on synthetic data; no downloads, training runs, or GPUs."""
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_utils import (EarlyStopping, cifar10_loaders, cifar10_train_val,
                         evaluate_classifier, split_dataset)


class FakeCIFAR10:
    def __init__(self, root, train, download, transform):
        self.train, self.transform = train, transform

    def __len__(self):
        return 100 if self.train else 20

    def __getitem__(self, index):
        return self.transform(Image.fromarray(np.full((32, 32, 3), index, dtype=np.uint8))), index % 10


def original_indices(dataset):
    if hasattr(dataset, 'indices'):
        parent = original_indices(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    return list(range(len(dataset)))


class EvaluationChecks(unittest.TestCase):
    @patch('torchvision.datasets.CIFAR10', FakeCIFAR10)
    def test_split_transforms_and_test_loader(self):
        config = timm.data.resolve_model_data_config('fastvit_t8')
        train_t = timm.data.create_transform(**config, is_training=True)
        eval_t = timm.data.create_transform(**config, is_training=False)
        train, val, test = cifar10_loaders(train_t, eval_t, 4, num_workers=0)
        self.assertEqual((len(train.dataset), len(val.dataset), len(test.dataset)), (80, 20, 20))
        self.assertTrue(set(train.dataset.indices).isdisjoint(val.dataset.indices))
        self.assertEqual(set(train.dataset.indices) | set(val.dataset.indices), set(range(100)))
        self.assertIs(train.dataset.dataset.transform, train_t)
        self.assertIs(val.dataset.dataset.transform, eval_t)
        self.assertIs(test.dataset.transform, eval_t)
        self.assertIsNot(val.dataset.dataset, train.dataset.dataset)
        self.assertFalse(test.dataset.train)
        self.assertIsNot(test.dataset, val.dataset)
        self.assertTrue(torch.equal(val.dataset[0][0], val.dataset[0][0]))
        self.assertNotIn('Random', repr(eval_t))
        repeat, _ = split_dataset(train.dataset.dataset)
        self.assertEqual(repeat.indices, train.dataset.indices)
        inner_train, inner_val = cifar10_train_val(train_t, eval_t, tuning=True)
        a, b = set(original_indices(inner_train)), set(original_indices(inner_val))
        self.assertTrue(a.isdisjoint(b))
        self.assertTrue((a | b).isdisjoint(val.dataset.indices))
        self.assertEqual(a | b, set(train.dataset.indices))
        self.assertIs(inner_val.dataset.dataset.transform, eval_t)

    def test_restores_independent_best_weights_and_buffers(self):
        model = torch.nn.BatchNorm1d(2)
        stop = EarlyStopping(patience=2)
        stop(1.0, model, 1)
        expected = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            model.weight.add_(10)
            model.running_mean.add_(10)
        stop(1.1, model, 2)
        stop(1.1, model, 3)
        self.assertTrue(stop.early_stop)
        stop.restore(model)
        self.assertEqual(stop.best_epoch, 1)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, expected[k]), k)
        with self.assertRaises(ValueError):
            stop(float('nan'), model, 4)

    def test_sample_weighted_loss_and_confusion_orientation(self):
        logits = torch.tensor([[5., 0.], [5., 0.], [0., 5.]])
        labels = torch.tensor([0, 1, 1])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = evaluate_classifier(torch.nn.Identity(), loader, 'cpu', num_classes=2)
        self.assertEqual(result['sample_count'], 3)
        self.assertEqual(result['correct'], 2)
        self.assertEqual(result['confusion_matrix'], [[1, 0], [1, 1]])
        self.assertAlmostEqual(result['loss'], torch.nn.functional.cross_entropy(logits, labels).item(), places=6)

    def test_training_restores_then_tests_once_before_saving(self):
        import contextlib
        import io
        import train
        from torch.utils.data import Subset

        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(2, 10)
            def reset_classifier(self, num_classes):
                pass
            def forward(self, inputs):
                return self.fc(inputs)

        data = TensorDataset(torch.randn(12, 2), torch.arange(12) % 10)
        loaders = [DataLoader(Subset(data, ids), batch_size=4)
                   for ids in [list(range(8)), [8, 9], [10, 11]]]
        model = TinyClassifier()
        stopper = EarlyStopping(patience=15)
        config = train.get_hyperparameters('fastvit_t8')
        config['batch_size'] = 4
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(train.timm, 'create_model', return_value=model))
            stack.enter_context(patch.object(train.timm.data, 'resolve_model_data_config', return_value={}))
            stack.enter_context(patch.object(train.timm.data, 'create_transform', return_value=None))
            stack.enter_context(patch.object(train, 'cifar10_loaders', return_value=loaders))
            stack.enter_context(patch.object(train, 'get_hyperparameters', return_value=config))
            stack.enter_context(patch.object(train, 'EarlyStopping', return_value=stopper))
            evaluate = stack.enter_context(patch.object(train, 'evaluate_classifier', wraps=evaluate_classifier))
            metadata = stack.enter_context(patch.object(train, 'save_model_metadata', return_value='synthetic'))
            saved = stack.enter_context(patch.object(train, 'save_model'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            train.train_model('fastvit_t8', device_name='cpu', epochs_override=6)
        evaluate.assert_called_once_with(model, loaders[2], torch.device('cpu'), 10)
        self.assertEqual(metadata.call_args.kwargs['metadata']['selected_epoch'], stopper.best_epoch)
        self.assertEqual(metadata.call_args.kwargs['metadata']['test_split'], 'CIFAR10(train=False)')
        saved.assert_called_once_with('synthetic', model)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, stopper.best_state[key]))

    def test_conversion_normalization(self):
        from convert_coreml import ImageClassifier
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        image = torch.rand(1, 3, 4, 4)
        wrapper = ImageClassifier(torch.nn.Identity(), mean, std)
        expected = ((image - torch.tensor(mean).view(1, 3, 1, 1)) /
                    torch.tensor(std).view(1, 3, 1, 1)).softmax(1)
        self.assertTrue(torch.allclose(wrapper(image), expected))


if __name__ == '__main__':
    unittest.main()
