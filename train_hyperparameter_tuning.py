import optuna
import torch
import torch.nn as nn
import timm
from train_utils import cifar10_train_val, select_device

# model_name = 'fastvit_t8.apple_dist_in1k'
# model_name = 'fastvit_t8'
# model_name = 'mobileone_s0'
# model_name = 'tiny_vit_21m_224'
model_name = None

device = select_device()

train_dataset, val_dataset = None, None


def get_hyperparameters_tunning_train_val_loaders(model_default_transforms):
    """Use 32k/8k inside the training partition; reserve final validation and test."""
    global train_dataset, val_dataset

    if train_dataset is not None and val_dataset is not None:
        print("Returning cached train_dataset and val_dataset")
        return train_dataset, val_dataset

    train_transforms, eval_transforms = model_default_transforms
    train_dataset, val_dataset = cifar10_train_val(
        train_transforms, eval_transforms, root=args.data_root, seed=args.seed,
        download=args.download, tuning=True,
    )

    return train_dataset, val_dataset


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2)
    scheduler_name = trial.suggest_categorical(
        "scheduler", ["StepLR", "ReduceLROnPlateau"]
    )

    print(f"[{model_name}] Trial: {trial.number}")

    # Load the model and dataset
    model = timm.create_model(model_name, pretrained=args.pretrained)
    data_config = timm.data.resolve_model_data_config(model)
    model_default_transforms = timm.data.create_transform(
        **data_config, is_training=False
    )
    train_transforms = timm.data.create_transform(**data_config, is_training=True)
    get_hyperparameters_tunning_train_val_loaders((train_transforms, model_default_transforms))

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
    for epoch in range(args.epochs):  # Use fewer epochs for hyperparameter optimization
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
                val_loss += criterion(outputs, labels).item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss /= len(val_dataset)
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
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--storage", default="sqlite:///build/tuning.sqlite3")
    parser.add_argument("--study-name", required=True, help="Use a distinct name for each model and initialization")
    args = parser.parse_args()
    import os
    os.makedirs("build", exist_ok=True)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.epochs < 1 or args.trials < 1:
        parser.error("--epochs and --trials must be positive")

    model_name = args.model

    # print(f"[{model_name}] Using device: {device}")

    # Run optimization
    study = optuna.create_study(
        direction="maximize", storage=args.storage, study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.set_user_attr("protocol", "corrected-inner-training-split-2026-11-23")
    study.set_user_attr("model", args.model)
    study.set_user_attr("pretrained", args.pretrained)
    study.set_user_attr("seed", args.seed)
    study.optimize(objective, n_trials=args.trials)
    print("Best accuracy and settings:", study.best_value, study.best_params)
