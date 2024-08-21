from ahcore.models.layers.MLP import MLP
from ahcore.models.layers.attention import GatedAttention

from typing import List, Optional

import torch
from torch import nn


class ABMIL(nn.Module):
    """
    Attention-based MIL (Multiple Instance Learning) classification model (See [1]_).
    This model is adapted from https://github.com/owkin/HistoSSLscaling/blob/main/rl_benchmarks/models/slide_models/abmil.py.
    It uses an attention mechanism to aggregate features from multiple instances (tiles) into a single prediction.

    Parameters
    ----------
    in_features : int
        Number of input features for each tile.
    out_features : int, optional
        Number of output features (typically 1 for binary classification), by default 1.
    attention_dimension : int, optional
        Dimensionality of the attention mechanism, by default 128.
    temperature : float, optional
        Temperature parameter for scaling the attention scores, by default 1.0.
    embed_mlp_hidden : Optional[List[int]], optional
        List of hidden layer sizes for the embedding MLP, by default None.
    embed_mlp_dropout : Optional[List[float]], optional
        List of dropout rates for the embedding MLP, by default None.
    embed_mlp_activation : Optional[torch.nn.Module], optional
        Activation function for the embedding MLP, by default nn.ReLU().
    embed_mlp_bias : bool, optional
        Whether to include bias in the embedding MLP layers, by default True.
    classifier_hidden : Optional[List[int]], optional
        List of hidden layer sizes for the classifier MLP, by default [128, 64].
    classifier_dropout : Optional[List[float]], optional
        List of dropout rates for the classifier MLP, by default None.
    classifier_activation : Optional[torch.nn.Module], optional
        Activation function for the classifier MLP, by default nn.ReLU().
    classifier_bias : bool, optional
        Whether to include bias in the classifier MLP layers, by default False.

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
        attention_dimension: int = 128,
        temperature: float = 1.0,
        embed_mlp_hidden: Optional[List[int]] = None,
        embed_mlp_dropout: Optional[List[float]] = None,
        embed_mlp_activation: Optional[torch.nn.Module] = nn.ReLU(),
        embed_mlp_bias: bool = True,
        classifier_hidden: Optional[List[int]] = [128, 64],
        classifier_dropout: Optional[List[float]] = None,
        classifier_activation: Optional[torch.nn.Module] = nn.ReLU(),
        classifier_bias: bool = False,
    ) -> None:
        """
        Initializes the ABMIL model with embedding and classification layers.

        Parameters
        ----------
        in_features : int
            Number of input features for each tile.
        out_features : int, optional
            Number of output features (typically 1 for binary classification), by default 1.
        attention_dimension : int, optional
            Dimensionality of the attention mechanism, by default 128.
        temperature : float, optional
            Temperature parameter for scaling the attention scores, by default 1.0.
        embed_mlp_hidden : Optional[List[int]], optional
            List of hidden layer sizes for the embedding MLP, by default None.
        embed_mlp_dropout : Optional[List[float]], optional
            List of dropout rates for the embedding MLP, by default None.
        embed_mlp_activation : Optional[torch.nn.Module], optional
            Activation function for the embedding MLP, by default nn.ReLU().
        embed_mlp_bias : bool, optional
            Whether to include bias in the embedding MLP layers, by default True.
        classifier_hidden : Optional[List[int]], optional
            List of hidden layer sizes for the classifier MLP, by default [128, 64].
        classifier_dropout : Optional[List[float]], optional
            List of dropout rates for the classifier MLP, by default None.
        classifier_activation : Optional[torch.nn.Module], optional
            Activation function for the classifier MLP, by default nn.ReLU().
        classifier_bias : bool, optional
            Whether to include bias in the classifier MLP layers, by default False.

        """
        super(ABMIL, self).__init__()

        self.embed_mlp = MLP(
            in_features=in_features,
            hidden=embed_mlp_hidden,
            bias=embed_mlp_bias,
            out_features=attention_dimension,
            dropout=embed_mlp_dropout,
            activation=embed_mlp_activation,
        )

        self.attention_layer = GatedAttention(dim=attention_dimension, temperature=temperature)

        self.classifier = MLP(
            in_features=attention_dimension,
            out_features=out_features,
            bias=classifier_bias,
            hidden=classifier_hidden,
            dropout=classifier_dropout,
            activation=classifier_activation,
        )

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the attention weights for the input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_tiles, in_features) representing the features of tiles.

        Returns
        -------
        torch.Tensor
            Attention weights for each tile.

        """
        tiles_emb = self.embed_mlp(x)
        attention_weights = self.attention_layer.attention(tiles_emb)
        return attention_weights

    def forward(
        self,
        features: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ABMIL model.

        Parameters
        ----------
        features : torch.Tensor
            Input tensor of shape (batch_size, n_tiles, in_features) representing the features of tiles.
        return_attention : bool, optional
            If True, also returns the attention weights, by default False.

        Returns
        -------
        torch.Tensor
            Logits representing the model's output.
        torch.Tensor, optional
            Attention weights, returned if return_attention is True.

        """
        tiles_emb = self.embed_mlp(features)  # BxN_tilesxN_features --> BxN_tilesx128
        scaled_tiles_emb, attention_weights = self.attention_layer(
            tiles_emb, return_attention_weights=True
        )  # BxN_tilesx128 --> Bx128
        logits = self.classifier(scaled_tiles_emb)  # Bx128 --> Bx1

        if return_attention_weights:
            return logits, attention_weights

        return logits
