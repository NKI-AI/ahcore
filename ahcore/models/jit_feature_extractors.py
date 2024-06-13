from pathlib import Path
from typing import Any

import torch
from torch.jit import ScriptModule, load

from ahcore.models.base_jit_model import BaseAhcoreJitModel
from ahcore.models.embedding_functions.vit_embed import ViTEmbed
from ahcore.utils.types import EmbedTokenNames


class DinoV2TokenNames(EmbedTokenNames):
    CLS_TOKEN_NAME = "x_norm_clstoken"
    PATCH_TOKEN_NAME = "x_norm_patchtokens"


class DinoV2JitModel(BaseAhcoreJitModel):
    """
    This class is a wrapper for the DinoV2 foundation model in Ahcore.
    """

    def __init__(self, model: ScriptModule, output_mode: str, cls_only_model: bool = False) -> None:
        """
        Constructor for the DinoV2 class.
        This can be a feature extractor based on very large pre-trained model.

        Parameters
        ----------
        model : ScriptModule
            The jit compiled model.

        output_mode : str
            Tells you about how to deal with cls and patch output

        cls_only_model : bool
            If True, we assume the model just returns a tensor with cls token

        Returns
        -------
        None
        """
        super().__init__(model=model, output_mode=output_mode)
        self._cls_only_model = cls_only_model
        self._set_forward_function()

    @classmethod
    def from_jit_path(cls, jit_path: Path, output_mode: str, cls_only_model: bool = False) -> Any:
        """
        Load a jit compiled model from a file path.

        Parameters
        ----------
        jit_path : Path
            The path to the jit compiled model.

        output_mode : OutputModeBase
            The output mode of the model. This is used to determine the forward function of the model.

        cls_only_model : bool
            If True, we assume the model just returns a tensor with cls token.

        Returns
        -------
        An instance of the AhcoreJitModel class.
        """
        model = load(jit_path)  # type: ignore
        return cls(model, output_mode, cls_only_model)

    def _set_forward_function(self) -> None:
        self._forward_function = ViTEmbed(self._output_mode).forward  # More complex forward function

    def forward(self, x: torch.Tensor) -> Any:
        if self._cls_only_model:
            cls_token = self._model(x)
            patch_tokens = None
        else:
            output = self._model(x)
            cls_token = output[DinoV2TokenNames.CLS_TOKEN_NAME]
            patch_tokens = output[DinoV2TokenNames.PATCH_TOKEN_NAME]
        return self._forward_function(cls_token=cls_token, patch_tokens=patch_tokens)
