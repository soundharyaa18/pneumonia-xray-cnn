# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Lightweight sequential CNN: 6 conv blocks -> global average pooling -> FC(128) -> FC(num_classes)."""

    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(96, 125, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(125, 150, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(150, 150, kernel_size=3, padding=1)  # added layer (ablation)
        # Global Average Pooling (instead of flattening large feature maps)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Dense layers
        self.fc1 = nn.Linear(150, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)  # (batch_size, 150)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)       # logits; use CrossEntropyLoss, softmax only at inference


def build_model(num_classes=4):
    return SimpleCNN(num_classes=num_classes)
