# this file includes the original nystrom attention and transmil model
# from https://github.com/lucidrains/nystrom-attention/blob/main/nystrom_attention/nystrom_attention.py
# and https://github.com/szc19990412/TransMIL/blob/main/models/TransMIL.py, respectively.

from typing import Any

import numpy as np
import torch
from torch import nn as nn

from ahcore.models.layers.attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer: type = nn.LayerNorm, dim: int = 512) -> None:
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, in_features: int = 1024, num_classes: int = 1, hidden_dimension: int = 512) -> None:
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=hidden_dimension)
        self._fc1 = nn.Sequential(nn.Linear(in_features, hidden_dimension), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dimension))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=hidden_dimension)
        self.layer2 = TransLayer(dim=hidden_dimension)
        self.norm = nn.LayerNorm(hidden_dimension)
        self._fc2 = nn.Linear(hidden_dimension, self.n_classes)

    def forward(self, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        h = features  # [B, n, in_features]

        h = self._fc1(h)  # [B, n, hidden_dimension]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, hidden_dimension]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, hidden_dimension]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, hidden_dimension]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, hidden_dimension]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits: torch.Tensor = self._fc2(h)  # [B, out_features]

        return logits
