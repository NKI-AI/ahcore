# encoding: utf-8
"""Different u-net implementation than unet.py"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def double_conv(in_channels, out_channels, dropout_rate=0.3, activation=nn.GELU()):
    """Basic double convolutional layer with activation and batch normalization."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.Dropout2d(dropout_rate),
        nn.BatchNorm2d(out_channels),
        activation,
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.Dropout2d(dropout_rate),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
        activation,
    )


def unet_downsample_layer(in_channels, out_channels, activation=nn.GELU()):
    """
    Basic UNet downsample layer with consisting of double convolutional layers followed
    with a 2d max pooling layer wrapped in nn.Sequential.
    """
    return nn.Sequential(nn.MaxPool2d(kernel_size=2), double_conv(in_channels, out_channels, activation=activation))


class UnetUpsampleLayer(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        _upsample: nn.Module | None = None
        if bilinear:
            _upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            )
        else:
            _upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.upsample = _upsample
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]
        half_diff_w = torch.div(diff_w, 2, rounding_mode="trunc")
        half_diff_h = torch.div(diff_h, 2, rounding_mode="trunc")
        x1 = F.pad(x1, [half_diff_w, diff_w - half_diff_w, half_diff_h, diff_h - half_diff_h])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Args:
        num_classes     : Number of output classes required
        num_input_ch  : Number of channels in input images (default 3)
        depth      : Number of layers in each side of U-net (default 5)
        num_initial_filters : Number of features in first layer (default 64)
        bilinear        : Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
        self,
        num_input_ch: int,
        num_classes: int,
        depth: int = 5,
        num_initial_filters: int = 128,
        bilinear: bool = False,
        apply_softmax_out: bool = False,
    ):

        # Check wether num_layers is more than zero.
        if depth < 1:
            raise ValueError(f"num_layers = {depth}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = depth
        self.num_classes = num_classes
        self.input_channels = num_input_ch
        self.hidden_features = num_initial_filters
        self.bilinear = bilinear
        self.apply_softmax_out = apply_softmax_out

        # Create layers of the UNet model.
        self.layers = self.create_unet()

    def create_unet(self):
        # Define a feature extractor.
        layers = [double_conv(self.input_channels, self.hidden_features)]

        # Define the down path.
        feats = self.hidden_features
        for _ in range(self.num_layers - 1):
            layers.append(unet_downsample_layer(feats, feats * 2))
            feats *= 2

        # Define the up path.
        for _ in range(self.num_layers - 1):
            layers.append(UnetUpsampleLayer(feats, feats // 2, self.bilinear))
            feats //= 2

        # Define the final classification layer.
        layers.append(nn.Conv2d(feats, self.num_classes, kernel_size=1))
        return nn.ModuleList(layers)

    def forward(self, x):
        x = x.float()

        # Feature extraction
        xi = [self.layers[0](x)]

        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        # Final classification layer
        out = self.layers[-1](xi[-1])

        # Apply softmax to output distribution over classes.
        if self.apply_softmax_out:
            out = torch.softmax(out, dim=1)
        return out
