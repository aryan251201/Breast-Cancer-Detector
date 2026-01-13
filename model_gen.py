import torch
import torch.nn as nn
import torch.nn.functional as F


class BreastCancerCNN(nn.Module):
    def __init__(self):
        super(BreastCancerCNN, self).__init__()

        # Feature extractor
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(128 * 28 * 28, 256)

        # Output heads
        self.level_head = nn.Linear(256, 4)      # Level 1–4
        self.type_head = nn.Linear(256, 2)       # Benign / Malignant

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        level_output = self.level_head(x)
        type_output = self.type_head(x)

        return level_output, type_output
