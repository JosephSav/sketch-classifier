from torch import nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self, num_classes, dropout=0.0):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x


