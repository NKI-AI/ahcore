from typing import Optional, List, Union, Tuple

import torch
from torch import nn

from ahcore.models.layers.MLP import MaskedLinear

"""Most of this stuff is adapted from utils from https://github.com/owkin/HistoSSLscaling/tree/main"""

class SelfAttention(nn.Module):
    """Multi-Head Self-Attention.

    Implementation adapted from https://github.com/rwightman/pytorch-image-models.

    Parameters
    ----------
    in_features : int
        Number of input features.

    num_heads : int = 8
        Number of attention heads. Should be an integer greater or equal to 1.

    qkv_bias : bool = False
        Whether to add a bias to the linear projection for query, key and value.

    attn_dropout : float = 0.0
        Dropout rate (applied before the multiplication with the values).

    proj_dropout : float = 0.0
        Dropout rate (applied after the multiplication with the values).
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        self.__build()

    def __build(self):
        """Build the `SelfAttention` module."""
        head_dim = self.in_features // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(
            self.in_features, self.in_features * 3, bias=self.qkv_bias
        )
        self.attn_drop = nn.Dropout(self.attn_dropout)
        self.proj = nn.Linear(self.in_features, self.in_features)
        self.proj_drop = nn.Dropout(self.proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, seq_len, in_features).

        Returns
        -------
        out : torch.Tensor
            Output tensor, shape (B, seq_len, in_features).
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GatedAttention(nn.Module):
    """Gated Attention, as defined in https://arxiv.org/abs/1802.04712.
    Permutation invariant Layer on dim 1.
    Parameters
    ----------
    d_model: int = 128
    temperature: float = 1.0
        Attention Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 1.0,
    ):
        super(GatedAttention, self).__init__()

        self.V = nn.Linear(d_model, d_model)
        self.U = nn.Linear(d_model, d_model)
        self.w = MaskedLinear(d_model, 1, "-inf")

        self.temperature = temperature

    def attention(
        self,
        features: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Gets attention logits.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        h_v = torch.tanh(self.U(features))

        u_v = torch.sigmoid(self.V(features))

        attention_logits = self.w(h_v * u_v, mask=mask) / self.temperature

        attention_weights = torch.softmax(attention_logits, 1)

        return attention_weights

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        h_v = torch.tanh(self.U(features))

        u_v = torch.sigmoid(self.V(features))

        attention_logits = self.w(h_v * u_v, mask=mask) / self.temperature

        attention_weights = torch.softmax(attention_logits, 1)
        # if not torch.any(attention_weights[mask]==0.0):
        #     raise RuntimeError(f"Masked indices got non-zero weight")

        # features = features.masked_fill(mask, float(0.0))
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), features)

        return scaled_attention.squeeze(1), attention_weights