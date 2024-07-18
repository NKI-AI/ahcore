"""This is a simple mode to upscale the features. Expects input of Bx1x1x768 and outputs BxKx224x224
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDecoderModel(nn.Module):
    def __init__(self, out_channels: int, in_channels: int = 768, conv_in_channels: int = 4) -> None:
        super().__init__()
        self._conv_in_channels = conv_in_channels
        # We have a fully connected layer to get a multiple of Mx7x7 for reshape to end up with Kx224x224
        self.fc = nn.Linear(in_channels, 7 * 7 * self._conv_in_channels)

        # Upscale Mx7x7 five times to get output of Kx224x224
        self.conv_trans1 = nn.ConvTranspose2d(self._conv_in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv_trans4 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.conv_trans5 = nn.ConvTranspose2d(8, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: Only working for Bx1x1x768. Not working for patch_embeddings
        x = x.view(x.size(0), -1)  # Flatten the input tensor from (B x 1 x 1 x 768) to (B, 768)
        x = F.relu(self.fc(x))
        x = x.view(-1, self._conv_in_channels, 7, 7)  # Reshape to number of conv_in_channels of 7x7
        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = F.relu(self.conv_trans3(x))
        x = F.relu(self.conv_trans4(x))
        x = self.conv_trans5(x)
        return x
