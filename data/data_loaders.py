import os
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


def get_data_loaders(
    data_root="/content/download/tiny-imagenet/tiny-imagenet-200",
    batch_size=32,
    train_subset_size=1000,
    val_subset_size=200,
    num_workers=2,
):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=os.path.join(data_root, "train"), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(data_root, "val"), transform=transform)

    # Sample randomly across all 200 classes instead of only the first 2-4 classes
    if train_subset_size is not None:
        g = torch.Generator().manual_seed(42)
        train_indices = torch.randperm(len(train_dataset), generator=g)[:train_subset_size].tolist()
        train_dataset = Subset(train_dataset, train_indices)

    if val_subset_size is not None:
        g = torch.Generator().manual_seed(42)
        val_indices = torch.randperm(len(val_dataset), generator=g)[:val_subset_size].tolist()
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader