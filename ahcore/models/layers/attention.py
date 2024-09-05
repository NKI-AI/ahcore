import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn


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
        dim: int = 128,
        temperature: float = 1.0,
    ):
        super(GatedAttention, self).__init__()

        self.V = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)
        self.w = nn.Linear(dim, 1)

        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Parameters
        ----------
        features: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        return_attention_weights: bool = False

        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        h_v = torch.tanh(self.U(features))

        u_v = torch.sigmoid(self.V(features))

        attention_logits = self.w(h_v * u_v) / self.temperature

        attention_weights = torch.softmax(attention_logits, 1)

        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), features)

        if return_attention_weights:
            return scaled_attention.squeeze(1), attention_weights

        return scaled_attention.squeeze(1)


def moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    eye = torch.eye(x.shape[-1], device=device)
    eye = rearrange(eye, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * eye - (xz @ (15 * eye - (xz @ (7 * eye - xz)))))

    return z


# main attention class


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        residual_conv_kernel: int = 33,
        eps: float = 1e-8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        h, m, iters, eps = self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if mask is not None:
                mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if mask is not None:
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l_dim = math.ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l_dim)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l_dim)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        if mask is not None:
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l_dim)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0
        else:
            divisor = torch.Tensor([l_dim]).to(q_landmarks.device)

        # masked mean (if mask exists)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities

        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = torch.einsum(einops_eq, q, k_landmarks)
        sim2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = torch.einsum(einops_eq, q_landmarks, k)

        # masking

        if mask is not None:
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out: torch.Tensor = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out
