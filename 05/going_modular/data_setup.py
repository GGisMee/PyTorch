import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(
    train_dir:str,
    test_dir:str,
    transform:transforms.Compose,
    batch_size:int,
    num_workers:int
    ) -> tuple[DataLoader, DataLoader, list]:
    """Creates a training and a testing DataLoader"""
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes
    train_dataloader = DataLoader(train_data, batch_size, True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size, False, num_workers=num_workers)
    return train_dataloader, test_dataloader, class_names
