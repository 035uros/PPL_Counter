# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrowdCounterCNN(nn.Module):
    def __init__(self):
        super(CrowdCounterCNN, self).__init__()

        # Convolutional layers to extract features from input image
        self.conv1 = nn.Conv2d(3, 16, kernel_size=9, padding=4)  # RGB -> 16 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, padding=3)  # 16 -> 32 filters
        self.conv3 = nn.Conv2d(32, 16, kernel_size=7, padding=3)  # 32 -> 16 filters
        self.conv4 = nn.Conv2d(16, 8, kernel_size=7, padding=3)  # 16 -> 8 filters

        # Final layer to predict density map (single channel)
        self.output_layer = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        # Pass input through layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Final density map
        x = self.output_layer(x)
        return x


# Quick test
if __name__ == "__main__":
    model = CrowdCounterCNN()
    dummy_input = torch.randn(1, 3, 480, 640)  # batch_size=1, RGB, H=480, W=640
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)  # should be (1, 1, H, W)
