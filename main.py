import torch
from models.costumnet import CustomNet
from eval import validate
from train import train

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


model = CustomNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 20
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)


print(f'Best validation accuracy: {best_acc:.2f}%')
