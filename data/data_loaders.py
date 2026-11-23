from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import Subset
import torch

transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# root/{classX}/x001.jpg

tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)
# Define subset sizes
train_subset_size = 1000  # Adjust as needed (e.g., 500 or 1000)
val_subset_size = 200

# Create subsets from original datasets
train_subset = Subset(tiny_imagenet_dataset_train, range(train_subset_size))
val_subset = Subset(tiny_imagenet_dataset_val, range(val_subset_size))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)