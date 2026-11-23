import torch
import torch.nn as nn
import timm
import os
from train_plots import save_model_metadata
from train_utils import (EarlyStopping, cifar10_loaders, evaluate_classifier,
                         index_digest, select_device)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import time



def get_hyperparameters(model_name, pretrained=False):
    """
    Retrieve hyperparameters for a given model.
    This function returns a dictionary of hyperparameters based on the model name provided.
    It currently supports hyperparameters for 'fastvit_t8' model.

    Parameters:
    model_name (str): The name of the model for which to retrieve hyperparameters.

    Returns:
    dict: A dictionary containing the hyperparameters 'learning_rate', 'weight_decay', and 'batch_size'.
    """
    if model_name.startswith("fastvit_t8"):
        return {
            "learning_rate": 1e-4 if pretrained else 1e-3,
            "weight_decay": 0.05,
            "batch_size": 256,
            "scheduler": "CosineAnnealingLR",
            "optimizer": "AdamW",
            "hyper_tunning_acc": "CorrectedPretrained" if pretrained else "CorrectedScratch",
            "num_epochs": 300,
        }

    if model_name.startswith("mobileone_s0"):
        # Historical hard-coded setting; trial provenance unavailable.
        return {
            "learning_rate": 0.0024622820877589764,
            "weight_decay": 7.987597578812497e-05,
            "batch_size": 16,
            "scheduler": "StepLR",
            "optimizer": "Adam",
            "hyper_tunning_acc": 5785,
        }

    # Default hyperparameters
    return {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "scheduler": "ReduceLROnPlateau",
        "optimizer": "Adam",
        "hyper_tunning_acc": "CorrectedDefault",
    }


def save_model(filename, model):
    """
    Save the state dictionary of a PyTorch model to a file.
    Args:
      filename (str): The name of the file (without extension) where the model's state dictionary will be saved.
      model (torch.nn.Module): The PyTorch model whose state dictionary is to be saved.
    Returns:
      None
    """
    os.makedirs("train_models/", exist_ok=True)

    full_filename = os.path.join("train_models", f"{filename}.pth")
    torch.save(model.state_dict(), full_filename)


def train_model(model_name, pretrained=False, seed=0, device_name="auto",
                root="./data", num_workers=2, download=False, epochs_override=None):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Training model: {model_name}, pretrained: {pretrained}...")

    ### Train Constants
    hyperparameters = get_hyperparameters(model_name, pretrained)

    num_epochs = (
        hyperparameters["num_epochs"] if "num_epochs" in hyperparameters else 500
    )
    if epochs_override is not None:
        num_epochs = epochs_override
    if num_epochs < 6:
        raise ValueError("Use at least 6 epochs for the five-epoch warmup")
    warmup_epochs = 5
    batch_size = hyperparameters["batch_size"]
    learning_rate = hyperparameters["learning_rate"]
    weight_decay = hyperparameters["weight_decay"]
    scheduler_name = hyperparameters["scheduler"]
    optimizer_name = hyperparameters["optimizer"]
    hyper_tunning_acc = hyperparameters["hyper_tunning_acc"]
    learning_rate_stepLR_step_size = 5
    early_stopping_patience = 15
    early_stopping_delta = 0

    print(f"_hyper{hyper_tunning_acc}")

    # Load the pre-trained model
    model = timm.create_model(model_name, pretrained=pretrained)

    data_config = timm.data.resolve_model_data_config(model)
    test_transforms = timm.data.create_transform(**data_config, is_training=False)
    train_transforms = timm.data.create_transform(**data_config, is_training=True)
    print(f"Train transforms: {train_transforms}")
    print(f"Validation transforms: {test_transforms}")

    ### Data

    train_loader, val_loader, test_loader = cifar10_loaders(
        train_transforms, test_transforms, batch_size, root, seed, num_workers, download
    )
    train_dataset, val_dataset, test_dataset = (
        train_loader.dataset, val_loader.dataset, test_loader.dataset
    )
    num_classes = 10
    model.reset_classifier(num_classes)
    print(
        f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}"
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    dataset_name = "CIFAR10"

    ### Model

    # Move the model to the appropriate device
    device = select_device(device_name)
    print(f"Using device: {device}")
    model.to(device)

    ### Training

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    scheduler = None
    # Set up scheduler
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=learning_rate_stepLR_step_size, gamma=0.1
        )
    elif scheduler_name == "CosineAnnealingLR":
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-6, total_iters=warmup_epochs
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs
        )

        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # Early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, delta=early_stopping_delta
    )

    def test_model():
        start_time = time.monotonic()
        result = evaluate_classifier(model, test_loader, device, num_classes)
        elapsed = time.monotonic() - start_time
        import numpy as np
        return (result["loss"], result["accuracy"], np.array(result["confusion_matrix"]),
                elapsed / result["sample_count"], elapsed)

    final_val_acc = 0
    losses_train = []
    losses_val = []
    accuracies_train = []
    accuracies_val = []
    epochs = []
    cm = None


    total_train_time = 0
    total_val_time = 0
    total_test_time = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("Training...")

        start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Print progress every batch
            print(
                f"Batch {batch_idx}/{len(train_loader)} - Loss: {running_loss / ((batch_idx + 1) * inputs.size(0)):.4f}",
                end="\r",
            )

        train_time = time.time() - start_time
        total_train_time += train_time

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.float() / len(train_dataset)
        losses_train.append(epoch_loss)
        accuracies_train.append(epoch_acc)
        epochs.append(epoch + 1)

        print(
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, Train Time: {train_time:.2f}s"
        )
        print("Validating...")

        start_time = time.time()

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Statistics
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)


                # Print progress every batch
                print(
                    f"Batch {batch_idx}/{len(val_loader)} - Loss: {val_running_loss / ((batch_idx + 1) * inputs.size(0)):.4f}",
                    end="\r",
                )

        val_time = time.time() - start_time
        total_val_time += val_time

        val_loss = val_running_loss / len(val_dataset)
        val_acc = val_running_corrects.float() / len(val_dataset)
        final_val_acc = val_acc.cpu().item()
        losses_val.append(val_loss)
        accuracies_val.append(val_acc)
        early_stopping(val_loss, model, epoch + 1)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}, Val Time: {val_time:.2f}s")

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Scheduler step
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

    print("Training complete, Total time: ", total_train_time + total_val_time, "s")

    early_stopping.restore(model)
    test_loss, test_acc, cm, inference_time, test_time = test_model()
    total_test_time += test_time
    total_time = total_train_time + total_val_time + total_test_time
    print(f"Selected epoch: {early_stopping.best_epoch}; true test accuracy: {test_acc:.4f}")

    model_filename = save_model_metadata(
        model_name=model_name,
        dataset_name=dataset_name,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        hyper_tunning_accuracy=hyper_tunning_acc,
        test_acc=test_acc,
        train_loss=losses_train,
        val_loss=losses_val,
        train_acc=[acc.cpu().item() for acc in accuracies_train],
        val_acc=[acc.cpu().item() for acc in accuracies_val],
        epochs=epochs,
        cm=cm,
        classes=classes,
        train_time=total_train_time,
        inference_time=inference_time,
        metadata={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": scheduler_name,
            "learning_rate_stepLR_step_size": learning_rate_stepLR_step_size,
            "batch_size": batch_size,
            "total_val_time": total_val_time,
            "total_test_time": total_test_time,
            "total_train_time": total_train_time,
            "total_time": total_time,
            "hyper_tunning_acc": hyper_tunning_acc,
            "final_val_acc": final_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "evaluation_protocol": "corrected-2026-11-23",
            "selected_epoch": early_stopping.best_epoch,
            "best_val_loss": early_stopping.val_loss_min,
            "seed": seed,
            "pretrained": pretrained,
            "pretrained_cfg": model.pretrained_cfg if pretrained else None,
            "torch_version": torch.__version__,
            "timm_version": timm.__version__,
            "data_config": data_config,
            "train_transform": str(train_transforms),
            "eval_transform": str(test_transforms),
            "train_indices_sha256": index_digest(train_dataset.indices),
            "val_indices_sha256": index_digest(val_dataset.indices),
            "test_split": "CIFAR10(train=False)",
            "optimizer": optimizer_name,
        },
    )
    save_model(model_filename, model)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A valid timm model name")
    parser.add_argument(
        "--model", metavar="path", required=True, help="The model name to fine tune"
    )
    parser.add_argument("--pretrained", action="store_true", help="Fine-tune timm pretrained weights; default: scratch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, help="Override epoch budget (minimum 6)")
    args = parser.parse_args()

    train_model(args.model, pretrained=args.pretrained, seed=args.seed,
                device_name=args.device, root=args.data_root, num_workers=args.num_workers,
                download=args.download, epochs_override=args.epochs)
