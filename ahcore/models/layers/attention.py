import torch
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
