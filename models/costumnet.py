import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256*28*28, 512) # 200 is the number of classes in TinyImageNet
        self.fc2 = nn.Linear(512, 200)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()


    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.pool(x)

        x = self.conv2(x).relu()
        x = self.pool(x)

        x = self.conv3(x).relu()
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x).relu()
        x = self.fc2(x)



        return x