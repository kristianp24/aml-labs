import torch
import torch.nn as nn
import wandb
from models.costumnet import CustomNet
from eval import validate
from train import train
from data.data_loaders import get_data_loaders

train_loader, val_loader = get_data_loaders(
    data_root="/content/download/tiny-imagenet/tiny-imagenet-200",
    batch_size=32,
)

# 1. Initialize W&B run
wandb.init(
    entity="kristian_prendi-politecnico-di-torino",
    project="tiny-imagenet-customnet",
    config={
        "architecture": "CustomNet",
        "optimizer": "SGD",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "num_epochs": 20,
    }
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = CustomNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=wandb.config.learning_rate,
    momentum=wandb.config.momentum
)

# Optional: track gradients and parameters
wandb.watch(model, log="gradients", log_freq=100)

best_acc = 0

# Run the training process for {num_epochs} epochs
for epoch in range(1, wandb.config.num_epochs + 1):
    train_loss, train_accuracy = train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_loss, val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)

    # 2. Log epoch metrics to W&B
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_accuracy,
        "val/loss": val_loss,
        "val/accuracy": val_accuracy,
        "val/best_accuracy": best_acc,
    })

print(f'Best validation accuracy: {best_acc:.2f}%')

# 3. Finish run cleanly
wandb.finish()