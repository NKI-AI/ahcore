from ahcore.models.layers.MLP import MLP, MaskedMLP
from ahcore.models.layers.attention import GatedAttention

from typing import List, Optional

import torch
from torch import nn
import torch.nn.init as init

# todo: fix docstring
class ABMIL(nn.Module):
    """Attention-based MIL classification model (See [1]_).
    Adapted from https://github.com/owkin/HistoSSLscaling/blob/main/rl_benchmarks/models/slide_models/abmil.py

    Example:
        >>> module = ABMIL(in_features=128, out_features=1)
        >>> logits, attention_scores = module(slide, mask=mask)
        >>> attention_scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    d_model_attention: int = 128
        Dimension of attention scores.
    temperature: float = 1.0
        GatedAttention softmax temperature.
    tiles_mlp_hidden: Optional[List[int]] = None
        Dimension of hidden layers in first MLP.
    mlp_hidden: Optional[List[int]] = None
        Dimension of hidden layers in last MLP.
    mlp_dropout: Optional[List[float]] = None,
        Dropout rate for last MLP.
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation for last MLP.
    bias: bool = True
        Add bias to the first MLP.

    References
    ----------
    .. [1] Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based
    deep multiple instance learning. In Jennifer Dy and Andreas Krause,
    editors, Proceedings of the 35th International Conference on Machine
    Learning, volume 80 of Proceedings of Machine Learning Research,
    pages 2127–2136. PMLR, 10–15 Jul 2018.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        number_of_tiles: int = 1000,
        d_model_attention: int = 128,
        temperature: float = 1.0,
        masked_mlp_hidden: Optional[List[int]] = None,
        masked_mlp_dropout: Optional[List[float]] = None,
        masked_mlp_activation: Optional[torch.nn.Module] = nn.Sigmoid(),
        mlp_hidden: Optional[List[int]] = [128, 64],
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = nn.Sigmoid(),
        bias: bool = True,
        use_positional_encoding: bool = False,
    ) -> None:
        super(ABMIL, self).__init__()

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length"
                    "as mlp_dropout if mlp_dropout is given."
                )

        self.embed_mlp = MLP(
            in_features=in_features,
            hidden=masked_mlp_hidden,
            bias=bias,
            out_features=d_model_attention,
            dropout=masked_mlp_dropout,
            activation=masked_mlp_activation,
        )

        self.attention_layer = GatedAttention(
            d_model=d_model_attention, temperature=temperature
        )

        mlp_in_features = d_model_attention

        self.mlp = MLP(
            in_features=mlp_in_features,
            out_features=out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

        self.use_positional_encoding = use_positional_encoding

        if self.use_positional_encoding:
            # TODO this should also add some interpolation
            self.positional_encoding = nn.Parameter(torch.zeros(1, number_of_tiles, in_features))
            init.trunc_normal_(self.positional_encoding, mean=0.0, std=0.02, a=-2.0, b=2.0)
            self.positional_encoding.requires_grad = True

    def interpolate_positional_encoding(self, coordinates: torch.Tensor):
        """
                Perform bilinear interpolation using the given coordinates on the positional encoding.
                The positional encoding is considered as a flattened array representing a (h, w) grid.

                Args:
                coordinates (torch.Tensor): The normalized coordinates tensor of shape (batch_size, 2),
                                            where each row is (x, y) in normalized coordinates [0, 1].

                Returns:
                torch.Tensor: The interpolated features from the positional encoding.
                """
        # Scale coordinates to the range of the positional encoding indices
        max_idx = int(torch.sqrt(torch.tensor([self.positional_encoding.shape[1]]))) - 1
        scaled_coordinates = max_idx * coordinates

        # Separate scaled coordinates into x and y components
        x = scaled_coordinates[..., 0]
        y = scaled_coordinates[..., 1]


        # Get integer parts of coordinates
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        # Clamp indices to ensure they remain within valid range
        x0 = torch.clamp(x0, 0, max_idx)
        x1 = torch.clamp(x1, 0, max_idx)
        y0 = torch.clamp(y0, 0, max_idx)
        y1 = torch.clamp(y1, 0, max_idx)

        # Calculate linear indices
        idx_q11 = y0 * max_idx + x0
        idx_q12 = y1 * max_idx + x0
        idx_q21 = y0 * max_idx + x1
        idx_q22 = y1 * max_idx + x1

        # Fetch the corner points
        q11 = self.positional_encoding[0, idx_q11, :]
        q12 = self.positional_encoding[0, idx_q12, :]
        q21 = self.positional_encoding[0, idx_q21, :]
        q22 = self.positional_encoding[0, idx_q22, :]

        # Compute fractional part for interpolation
        x_frac = x - x0.float()
        y_frac = y - y0.float()

        # Bilinear interpolation
        interpolated_positional_encoding = (q11 * (1 - x_frac).unsqueeze(2) * (1 - y_frac).unsqueeze(2) +
                               q12 * (1 - x_frac).unsqueeze(2) * y_frac.unsqueeze(2) +
                               q21 * x_frac.unsqueeze(2) * (1 - y_frac).unsqueeze(2) +
                               q22 * x_frac.unsqueeze(2) * y_frac.unsqueeze(2))

        return interpolated_positional_encoding

    def get_attention(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None, coordinates: torch.Tensor = None,
    ) -> torch.Tensor:
        """Get attention logits.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        if self.use_positional_encoding:
            positional_encoding = self.interpolate_positional_encoding(coordinates)
            x = x + positional_encoding

        tiles_emb = self.tiles_emb(x, mask)
        attention_weights = self.attention_layer.attention(tiles_emb, mask)
        return attention_weights

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None, coordinates: torch.Tensor = None, return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        coordinates
        features: torch.Tensor
            (B, N_TILES, D+3)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, OUT_FEATURES), (B, N_TILES)
        """
        if coordinates is None and self.positional_encoding:
            raise ValueError(f"Coordinates of NoneType are not accepted if positional_encoding is used")

        if self.use_positional_encoding:
            positional_encoding = self.interpolate_positional_encoding(coordinates)
            features = features + positional_encoding

        tiles_emb = self.embed_mlp(features)  # BxN_tilesxN_features --> BxN_tilesx128
        scaled_tiles_emb, attention_weights = self.attention_layer(tiles_emb, mask) # BxN_tilesx128 --> Bx128
        logits = self.mlp(scaled_tiles_emb)  # Bx128 --> Bx1

        if return_attention:
            return logits, attention_weights

        return logits