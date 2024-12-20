import torch
from torch.utils.data import random_split


def split_dataset(full_dataset, train_ratio=0.8, seed=0):
    """
    Splits a dataset into training and validation sets.
    Args:
      full_dataset (Dataset): The complete dataset to be split.
      train_ratio (float, optional): The ratio of the dataset to be used for training.
                       Default is 0.8.
      seed (int, optional): The seed for the random number generator to ensure reproducibility.
                  Default is 0.
    Returns:
      tuple: A tuple containing the training dataset and the validation dataset.
    """

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=gen
    )
    return train_dataset, val_dataset
