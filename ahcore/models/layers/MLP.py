from typing import Optional, List, Union, Tuple

import torch
from torch import nn

"""Most of this stuff is adapted from utils from https://github.com/owkin/HistoSSLscaling/tree/main"""


class MLP(nn.Sequential):
    """MLP Module.

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    hidden: Optional[List[int]] = None
        Dimension of hidden layer(s).
    dropout: Optional[List[float]] = None
        Dropout rate(s).
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        MLP activation.
    bias: bool = True
        Add bias to MLP hidden layers.

    Raises
    ------
    ValueError
        If ``hidden`` and ``dropout`` do not share the same length.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[nn.Module] = nn.Sigmoid(),
        bias: bool = True,
    ):
        if dropout is not None:
            if hidden is not None:
                assert len(hidden) == len(
                    dropout
                ), "hidden and dropout must have the same length"
            else:
                raise ValueError(
                    "hidden must have a value and have the same length as dropout if dropout is given."
                )

        d_model = in_features
        layers = []

        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [nn.Linear(d_model, h, bias=bias)]
                d_model = h

                if activation is not None:
                    seq.append(activation)

                if dropout is not None:
                    seq.append(nn.Dropout(dropout[i]))

                layers.append(nn.Sequential(*seq))

        layers.append(nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)

class MaskedLinear(nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.
    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.mask_value = mask_value

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):  # pylint: disable=arguments-renamed
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, SEQ_LEN, IN_FEATURES).
        mask: Optional[torch.BoolTensor] = None
            True for values that were padded, shape (B, SEQ_LEN, 1),

        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mask_value={self.mask_value}, bias={self.bias is not None}"
        )


class MaskedMLP(nn.Module):
    """MLP to be applied to tiles to compute scores.
    This module can be used in combination of a mask
    to prevent padding from influencing the scores values.
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    hidden: Optional[List[int]] = None
        Number of hidden layers and their respective number of features.
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    activation: torch.nn.Module = torch.nn.Sigmoid()
        MLP activation function
    dropout: Optional[torch.nn.Module] = None
        Optional dropout module. Will be interlaced with the linear layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden: Optional[List[int]] = None,
        bias: bool = True,
        activation: nn.Module = nn.Sigmoid(),
        dropout: Optional[nn.Module] = None,
    ):
        super(MaskedMLP, self).__init__()

        if dropout is not None:
            assert len(dropout) == len(hidden), "Length of dropout is not correct"

        self.hidden_layers = nn.ModuleList()
        if hidden is not None:
            for i, h in enumerate(hidden):
                self.hidden_layers.append(
                    MaskedLinear(in_features, h, bias=bias, mask_value="-inf")
                )
                self.hidden_layers.append(activation)
                if dropout:
                    self.hidden_layers.append(nn.Dropout(dropout[i]))
                in_features = h

        self.hidden_layers.append(
            nn.Linear(in_features, out_features, bias=bias)
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES), True for values that were padded.

        Returns
        -------
        x: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x_before = x.clone().detach()
                x = layer(x)

            if torch.any(x.masked_fill(mask, 0).isnan()):
                raise RuntimeError(f"Found NaN values in x outside the mask")

        return x

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