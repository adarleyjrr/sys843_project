import optuna
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import timm
from torch.utils.data import random_split
from train_utils import split_dataset

# model_name = 'fastvit_t8.apple_dist_in1k'
# model_name = 'fastvit_t8'
# model_name = 'mobileone_s0'
# model_name = 'tiny_vit_21m_224'
model_name = None

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_dataset, val_dataset = None, None


def get_hyperparameters_tunning_train_val_loaders(model_default_transforms):
    global train_dataset, val_dataset

    if train_dataset is not None and val_dataset is not None:
        print("Returning cached train_dataset and val_dataset")
        return train_dataset, val_dataset

    full_train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=model_default_transforms
    )
    # Split the 50000 train dataset into 40000 training and 10000 validation, with a fixed seed for consistency between trials
    _, full_val_dataset = split_dataset(full_train_dataset)

    # Split the full_val_dataset into train_dataset (80%) and val_dataset (20%)
    train_dataset, val_dataset = split_dataset(full_val_dataset)

    return train_dataset, val_dataset


def objective(trial):
    # Suggest hyperparameters
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    # weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2)
    # scheduler_name = trial.suggest_categorical(
    #     "scheduler", ["StepLR", "ReduceLROnPlateau"]
    # )

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2)
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["StepLR", "ReduceLROnPlateau"]
    )

    print(f"[{model_name}] Trial: {trial.number}")

    # Load the model and dataset
    model = timm.create_model(model_name, pretrained=False)
    data_config = timm.data.resolve_model_data_config(model)
    model_default_transforms = timm.data.create_transform(
        **data_config, is_training=False
    )
    get_hyperparameters_tunning_train_val_loaders(model_default_transforms)

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

    # Prepare data loaders with suggested batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Reset the model
    num_classes = len(classes)
    model.reset_classifier(num_classes)

    # Move the model to the appropriate device
    # For some reason if I transfer to device right after creating the model, it doesn't work
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] Using device: {device}")
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Set up optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )

    # Set up scheduler
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # Training loop (simplified)
    for epoch in range(10):  # Use fewer epochs for hyperparameter optimization
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / len(val_dataset)

        # Scheduler step
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Report intermediate objective value
        trial.report(val_acc, epoch)

        # Handle pruning (optional)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc  # Objective to maximize


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A valid timm model name")
    parser.add_argument(
        "--model", metavar="path", required=True, help="The model name to fine tune"
    )
    args = parser.parse_args()

    model_name = args.model

    # print(f"[{model_name}] Using device: {device}")

    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
