"""Split, checkpoint selection, and evaluation shared by the corrected workflow."""
import copy
import hashlib
import math

import torch
from torch.utils.data import DataLoader, Subset, random_split


def split_dataset(full_dataset, train_ratio=0.8, seed=0, eval_dataset=None):
    """Split indices reproducibly; optionally use a separate evaluation dataset view.

    Both views must have identical example order and labels. This keeps random
    training augmentation out of validation without changing split membership.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if eval_dataset is not None and len(eval_dataset) != len(full_dataset):
        raise ValueError("Training and evaluation views must have equal length")
    train_size = int(train_ratio * len(full_dataset))
    train, val = random_split(
        full_dataset, [train_size, len(full_dataset) - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    if eval_dataset is not None:
        val = Subset(eval_dataset, val.indices)
    return train, val


def cifar10_train_val(train_transform, eval_transform, root="./data", seed=0,
                      download=False, tuning=False):
    from torchvision.datasets import CIFAR10

    augmented = CIFAR10(root=root, train=True, download=download, transform=train_transform)
    deterministic = CIFAR10(root=root, train=True, download=download, transform=eval_transform)
    train, val = split_dataset(augmented, seed=seed, eval_dataset=deterministic)
    if tuning:
        # Inner split of the 40k training partition. Final validation stays untouched.
        eval_train = Subset(deterministic, train.indices)
        train, val = split_dataset(train, seed=seed, eval_dataset=eval_train)
    return train, val


def cifar10_loaders(train_transform, eval_transform, batch_size, root="./data",
                    seed=0, num_workers=2, download=False):
    from torchvision.datasets import CIFAR10

    train, val = cifar10_train_val(train_transform, eval_transform, root, seed, download)
    test = CIFAR10(root=root, train=False, download=download, transform=eval_transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                   generator=torch.Generator().manual_seed(seed)),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def index_digest(indices):
    return hashlib.sha256(
        ','.join(str(i) for i in indices).encode('ascii')
    ).hexdigest()


def select_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class EarlyStopping:
    """Select by validation loss, retaining an independent CPU copy of the weights."""
    def __init__(self, patience=5, delta=0):
        self.patience, self.delta = patience, delta
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.best_state = None
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):
        if not math.isfinite(val_loss):
            raise ValueError("Non-finite validation loss")
        if val_loss < self.val_loss_min - self.delta:
            self.val_loss_min = val_loss
            self.best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience

    def restore(self, model):
        if self.best_state is None:
            raise RuntimeError("No validation checkpoint has been selected")
        model.load_state_dict(self.best_state)


def evaluate_classifier(model, loader, device, num_classes=10):
    """Return sample-weighted loss and confusion matrix (true rows, predicted columns)."""
    model.eval()
    loss_sum, count, correct = 0.0, 0, 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss_sum += torch.nn.functional.cross_entropy(outputs, labels, reduction="sum").item()
            preds = outputs.argmax(dim=1)
            count += labels.numel()
            correct += (preds == labels).sum().item()
            cm += torch.bincount(
                (labels * num_classes + preds).cpu(), minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)
    if count == 0:
        raise ValueError("Empty evaluation loader")
    return {"sample_count": count, "correct": correct, "accuracy": correct / count,
            "loss": loss_sum / count, "confusion_matrix": cm.tolist()}
