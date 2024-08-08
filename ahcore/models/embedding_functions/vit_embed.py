from typing import Optional

import torch

from ahcore.models.embedding_functions.base_embedding import AhCoreFeatureEmbedding
from ahcore.utils.types import ViTEmbedMode


class ViTEmbed(AhCoreFeatureEmbedding):
    def __init__(self, embedding_mode: ViTEmbedMode):
        super().__init__()
        self._embedding_mode = embedding_mode
        self._set_embed_function()

    @property
    def dim_factor(self) -> int:
        """
        Returns the scaling factor by which the output feature dimensionality will increase
        when using a certain embedding method.
        E.g. the concat method will make the output dimensionality twice as big.
        """
        if self._embedding_mode == ViTEmbedMode.CONCAT:
            return 2
        else:
            return 1

    def _set_embed_function(self) -> None:
        if self._embedding_mode == ViTEmbedMode.CLS_ONLY:
            self.embed_fn = self.embed_cls_only
        elif self._embedding_mode == ViTEmbedMode.PATCH_ONLY:
            self.embed_fn = self.embed_patch_only
        elif self._embedding_mode == ViTEmbedMode.MEAN:
            self.embed_fn = self.embed_mean
        elif self._embedding_mode == ViTEmbedMode.CONCAT_MEAN:
            self.embed_fn = self.embed_concat_mean
        elif self._embedding_mode == ViTEmbedMode.CONCAT:
            self.embed_fn = self.embed_concat
        else:
            raise NotImplementedError(f"Embedding mode {self._embedding_mode} is not supported.")

    def forward(self, cls_token: torch.Tensor, patch_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Returns vision transformer embedding.

        Parameters
        ----------
        cls_token :
            class token of shape (B, feature_dim)
        patch_tokens :
            patch tokens of (B, num_patches, feature_dim)

        Returns
        -------
        output: torch.Tensor
            Embedding of shape [B, feature_dim]

        """
        output = self.embed_fn(cls_token, patch_tokens)
        return output

    @staticmethod
    def embed_cls_only(cls_token: torch.Tensor, patch_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        return cls_token

    @staticmethod
    def embed_patch_only(cls_token: torch.Tensor, patch_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if patch_tokens is None:
            raise ValueError("patch_tokens cannot be None for this embedding mode")
        return patch_tokens.mean(dim=1)

    @staticmethod
    def embed_mean(cls_token: torch.Tensor, patch_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if patch_tokens is None:
            raise ValueError("patch_tokens cannot be None for this embedding mode")
        cls_token = cls_token.unsqueeze(1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        output = tokens.mean(dim=1)
        return output

    @staticmethod
    def embed_concat_mean(cls_token: torch.Tensor, patch_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if patch_tokens is None:
            raise ValueError("patch_tokens cannot be None for this embedding mode")
        cls_token = cls_token.unsqueeze(1)
        mean_patch_token = patch_tokens.mean(dim=1).unsqueeze(1)
        tokens = torch.cat([cls_token, mean_patch_token], dim=1)
        output = tokens.mean(dim=1)
        return output

    @staticmethod
    def embed_concat(cls_token: torch.Tensor, patch_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if patch_tokens is None:
            raise ValueError("patch_tokens cannot be None for this embedding mode")
        mean_patch_token = patch_tokens.mean(dim=1)
        output = torch.cat([cls_token, mean_patch_token], dim=1)
        return output
