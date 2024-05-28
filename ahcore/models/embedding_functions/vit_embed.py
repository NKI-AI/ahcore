from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable

import torch

from ahcore.utils.types import OutputModeBase, ViTEmbedMode


class _TokenNames(str, Enum):
    def __str__(self) -> Any:
        return self.value


class ViTTokenNames(_TokenNames):
    CLS_TOKEN_NAME = "x_norm_clstoken"
    PATCH_TOKEN_NAME = "x_norm_patchtokens"


class AhCoreFeatureEmbedding(ABC):
    def __init__(self, embedding_mode: OutputModeBase):
        self._embedding_mode = embedding_mode
        self.embed_fn: Callable[[dict[str, Any]], torch.Tensor]

    @property
    def embedding_mode(self) -> Any:
        return self._embedding_mode.value

    @property
    @abstractmethod
    def token_names(self) -> Any:
        raise NotImplementedError

    def _set_embed_function(self) -> None:
        raise NotImplementedError


class ViTEmbed(AhCoreFeatureEmbedding):
    def __init__(self, embedding_mode: OutputModeBase):
        super().__init__(embedding_mode=embedding_mode)
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

    @property
    def token_names(self) -> tuple[ViTTokenNames, ViTTokenNames]:
        return ViTTokenNames.CLS_TOKEN_NAME, ViTTokenNames.PATCH_TOKEN_NAME

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

    @staticmethod
    def get_output_tokens(lit_module_prediction_sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        cls_token = lit_module_prediction_sample[ViTTokenNames.CLS_TOKEN_NAME]
        patch_tokens = lit_module_prediction_sample[ViTTokenNames.PATCH_TOKEN_NAME]
        return cls_token, patch_tokens

    def embed_cls_only(self, lit_module_prediction_sample: dict[str, Any]) -> torch.Tensor:
        cls_token, _ = self.get_output_tokens(lit_module_prediction_sample)
        return cls_token

    def embed_patch_only(self, lit_module_prediction_sample: dict[str, Any]) -> torch.Tensor:
        _, patch_tokens = self.get_output_tokens(lit_module_prediction_sample)
        return patch_tokens

    def embed_mean(self, lit_module_prediction_sample: dict[str, Any]) -> torch.Tensor:
        cls_token, patch_tokens = self.get_output_tokens(lit_module_prediction_sample)
        cls_token = cls_token.unsqueeze(1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        output = tokens.mean(dim=1)
        return output

    def embed_concat_mean(self, lit_module_prediction_sample: dict[str, Any]) -> torch.Tensor:
        cls_token, patch_tokens = self.get_output_tokens(lit_module_prediction_sample)
        cls_token = cls_token.unsqueeze(1)
        mean_patch_token = patch_tokens.mean(dim=1).unsqueeze(1)
        tokens = torch.cat([cls_token, mean_patch_token], dim=1)
        output = tokens.mean(dim=1)
        return output

    def embed_concat(self, lit_module_prediction_sample: dict[str, Any]) -> torch.Tensor:
        cls_token, patch_tokens = self.get_output_tokens(lit_module_prediction_sample)
        mean_patch_token = patch_tokens.mean(dim=1)
        output = torch.cat([cls_token, mean_patch_token], dim=1)
        return output
