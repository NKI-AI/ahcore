from typing import List

import torch
from torch import nn as nn


class ConvBlock(nn.Module):
    """ConvBlock"""

    def __init__(self, ch_in: int, ch_out: int, dropout_rate: float):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), padding=1),
            nn.Dropout2d(dropout_rate),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), padding=1),
            nn.Dropout2d(dropout_rate),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        """forward"""
        x = self.conv(x)
        return x


class DeconvBlock(nn.Module):
    """DeconvBlock"""

    def __init__(self, ch_in: int, ch_out: int, dropout_rate: float):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            (nn.ConvTranspose2d(ch_in, ch_out, kernel_size=(2, 2), stride=(2, 2))),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x) -> torch.Tensor:
        """forward"""
        x = self.deconv(x)
        return x


class AttentionBlock(nn.Module):
    """AttentionBlock"""

    def __init__(self, ch_in: int, ch_skip: int, ch_out: int, dropout_rate: float):
        super(AttentionBlock, self).__init__()
        self.W_skip = nn.Sequential(
            nn.Conv2d(ch_skip, ch_out, kernel_size=(1, 1), padding=0),
            nn.Dropout2d(dropout_rate),
            nn.BatchNorm2d(ch_out),
        )

        self.W_in = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), padding=0),
            nn.Dropout2d(dropout_rate),
            nn.BatchNorm2d(ch_out),
        )

        self.relu = nn.ReLU()

        self.psi = nn.Sequential(
            nn.Conv2d(ch_out, 1, kernel_size=(1, 1), padding=0),
            nn.Dropout2d(dropout_rate),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """forward"""
        g = self.W_skip(skip)
        x = self.W_in(x)
        psi = self.relu(g + x)
        psi = self.psi(psi)
        return skip * psi


class Encoder(nn.Module):
    """Encoder"""

    def __init__(self, inp_channel: int, kernel_multiplier: int, dropout_rate: float, depth: int):
        super().__init__()
        self.depth = depth
        for i in range(self.depth + 1):
            ch_in = inp_channel if i == 0 else ch_out
            ch_out = kernel_multiplier * (2 ** (i + 3))
            setattr(self, f"conv_{i}", ConvBlock(ch_in, ch_out, dropout_rate))
            if i < self.depth:
                setattr(self, f"pool_{i}", nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

    def forward(self, x) -> List:
        """forward"""
        enc_features = []
        for i in range(self.depth):
            x = getattr(self, f"conv_{i}")(x)
            enc_features.append(x)
            x = getattr(self, f"pool_{i}")(x)
        x = getattr(self, f"conv_{self.depth}")(x)
        enc_features.append(x)
        return enc_features


class Decoder(nn.Module):
    """Decoder"""

    def __init__(
        self,
        kernel_multiplier: int,
        dropout_rate: float,
        num_classes: int,
        depth: int,
    ):
        super().__init__()
        self.depth = depth
        for i in range(self.depth):
            ch_in = kernel_multiplier * (2 ** (self.depth - (i + 1) + 4))
            ch_out = kernel_multiplier * (2 ** (self.depth - (i + 1) + 3))
            next_lvl_ch = kernel_multiplier * (2 ** (self.depth - (i + 1) + 2))
            setattr(self, f"up_sample_{i}", DeconvBlock(ch_in, ch_out, dropout_rate))
            setattr(
                self,
                f"attention_{i}",
                AttentionBlock(ch_in=ch_out, ch_skip=ch_out, ch_out=next_lvl_ch, dropout_rate=dropout_rate),
            )
            setattr(self, f"dec_conv_{i}", ConvBlock(ch_in=ch_in, ch_out=ch_out, dropout_rate=dropout_rate))

        self.output_layer = OutputLayer(kernel_multiplier * 8, num_classes)

    def forward(self, encoder_features: List) -> torch.Tensor:
        """forward"""
        dec_features = []
        for i in range(self.depth):
            dec_feature = getattr(self, f"up_sample_{i}")(encoder_features[-i - 1])
            attention_maps = getattr(self, f"attention_{i}")(dec_feature, encoder_features[-i - 2])
            dec_feature = torch.cat([dec_feature, attention_maps], dim=1)
            dec_feature = getattr(self, f"dec_conv_{i}")(dec_feature)
            dec_features.append(dec_feature)
        return self.output_layer(dec_features[-1])


class OutputLayer(nn.Module):
    """OutputLayer"""

    def __init__(self, inp_ch: int, num_classes: int):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=inp_ch,
                out_channels=num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
            )
        ]
        self.output_layer = nn.Sequential(*layers)

    def __call__(self, x):
        return self.output_layer(x)


class AttentionUnet(nn.Module):
    """AttentionUnet"""

    def __init__(self, inp_channel: int, kernel_multiplier: int, depth: int, dropout_rate: float, num_classes: int):
        super().__init__()
        self.encoder = Encoder(inp_channel, kernel_multiplier, dropout_rate, depth=depth)
        self.decoder = Decoder(kernel_multiplier, dropout_rate, num_classes, depth=depth)

    def forward(self, x):
        """
        Args:
            x: (batch_size, inp_channel, height, width)
        Returns:
            output: (batch_size, num_classes, height, width)
        """
        encoder_features = self.encoder(x)
        return self.decoder(encoder_features)
