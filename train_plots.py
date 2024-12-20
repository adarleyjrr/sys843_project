import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def save_model_metadata(
    model_name,
    dataset_name,
    train_size,
    val_size,
    hyper_tunning_accuracy,
    test_acc,
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    epochs,
    cm,
    classes,
    train_time,
    inference_time,
    metadata,
):
    """
    Saves the plot data to a pickle file.

    Parameters:
    model_name (str): The name or description of the model being trained.
    dataset_name (str): The name of the dataset.
    train_size (int): The size of the training dataset.
    val_size (int): The size of the validation dataset.
    hyper_tunning_accuracy (int): The decimals of accuracy of the model after hyperparameter tuning.
    test_acc (float): The accuracy of the model on the test dataset.
    train_loss (list or array-like): List or array of training loss values.
    val_loss (list or array-like): List or array of validation loss values.
    train_acc (list or array-like): List or array of training accuracy values.
    val_acc (list or array-like): List or array of validation accuracy values.
    epochs (list or array-like): List or array of epoch numbers.
    cm (array-like): Confusion matrix.
    classes (list): List of class names.
    train_time (float): Training time, not including validation nor test.
    inference_time (float): Inference time.
    metadata (dict): Additional metadata to be saved along with the plot.

    Returns:
    str: The filename of the saved pickle file.
    """
    data = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "train_size": train_size,
        "val_size": val_size,
        # "hyper_tunning_accuracy": hyper_tunning_accuracy, # Not needed
        "test_acc": test_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "epochs": epochs,
        "metadata": metadata,
        "confusion_matrix": cm,
        "classes": classes,
        "train_time": train_time,
        "inference_time": inference_time,
    }
    last_epoch = epochs[-1]

    # Create the directory if it doesn't exist
    os.makedirs("train_metadata/", exist_ok=True)

    existing_files = os.listdir("train_metadata/")
    last_order = 0
    if existing_files:
        order_numbers = [
            int(f.split("-")[0]) for f in existing_files if f.split("-")[0].isdigit()
        ]
        if order_numbers:
            last_order = max(order_numbers)
    order = last_order + 1
    order = f"{order:03d}"

    # Create the base filename
    base_filename = f"{order}-{model_name}_{dataset_name}_train{train_size}_val{val_size}_epochs{last_epoch}_hyper{hyper_tunning_accuracy}"
    full_filename = os.path.join("train_metadata", base_filename + ".pkl")

    # Check if the file already exists and add an order number if necessary
    if os.path.exists(full_filename):
        raise FileExistsError(f"File {full_filename} already exists.")

    with open(full_filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Metadata saved to {full_filename}")

    return base_filename


def load_and_plot_all_data(pickle_file):
    """
    Loads data from a pickle file and plots all relevant plots.

    Parameters:
    pickle_file (str): The path to the pickle file containing the data.

    Returns:
    None
    """
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    # Print data that isn't plotted
    print("Model Name:", data["model_name"])
    print("Dataset Name:", data["dataset_name"])
    print("Training Size:", data["train_size"])
    print("Validation Size:", data["val_size"])
    print("Epochs:", data["epochs"][-1])
    if "test_acc" in data:
        print("Test Accuracy:", f"{data["test_acc"] * 100:.2f}%")
    else:
        print("Validation Accuracy:", f"{data["val_acc"][-1] * 100:.2f}%")
    print("Training Time:", f"{data["train_time"]:.2f} seconds")
    print("Inference Time:", f"{data["inference_time"] * 1000:.2f} ms")
    print("Additional Metadata:", data["metadata"])

    # Create a figure for loss and accuracy
    plt.figure(figsize=(16, 8))

    # Subplot for loss curves
    plt.subplot(1, 2, 1)
    plt.plot(data["epochs"], data["train_loss"], label="Train Loss")
    plt.plot(data["epochs"], data["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Train and Validation Loss - {data['model_name']}")
    plt.legend()
    plt.grid(True)

    # Subplot for accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(data["epochs"], data["train_acc"], label="Train Accuracy")
    plt.plot(data["epochs"], data["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Train and Validation Accuracy - {data['model_name']}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Create a separate figure for confusion matrix
    plt.figure(figsize=(8, 8))
    cm = data["confusion_matrix"]
    classes = data["classes"]
    # cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {data['model_name']}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def list_and_plot_metadata():
    """
    Lists files in /train_metadata directory, allows user to pick one, and invokes load_and_plot_all_data.
    Once user is done, allows to pick again from the list, or quit.

    Returns:
    None
    """
    metadata_dir = "train_metadata"

    while True:
        # List all files in the directory
        files = [f for f in os.listdir(metadata_dir) if f.endswith(".pkl")]
        if not files:
            print("No metadata files found in the directory.")
            return

        files.sort()
        print("Available metadata files:")
        for i, file in enumerate(files):
            print(f"{i + 1}. {file}")

        # Ask user to pick a file or quit
        choice = input(
            "Enter the number of the file to load and plot (or 'q' to quit): "
        ).strip()
        if choice.lower() == "q":
            break

        try:
            file_index = int(choice) - 1
            if file_index < 0 or file_index >= len(files):
                print("Invalid choice. Please try again.")
                continue

            selected_file = os.path.join(metadata_dir, files[file_index])
            load_and_plot_all_data(selected_file)

        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


if __name__ == "__main__":
    list_and_plot_metadata()
