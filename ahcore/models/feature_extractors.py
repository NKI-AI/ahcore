from typing import Any

import torch
from torch.jit import ScriptModule

from ahcore.models.embedding_functions.vit_embed import ViTEmbed
from ahcore.models.jit_model import AhcoreJitModel
from ahcore.utils.types import OutputModeBase


class DinoV2JitModel(AhcoreJitModel):
    """
    This class is a wrapper for the DinoV2 foundation model in Ahcore.
    """

    def __init__(self, model: ScriptModule, embed_mode: OutputModeBase) -> None:
        """
        Constructor for the DinoV2 class.
        This can be a feature extractor based on very large pre-trained model.

        Parameters
        ----------
        model : ScriptModule
            The jit compiled model.

        embed_mode : ViTEmbedMode
            The output mode of the model. This is used to determine the forward function of the model.

        Returns
        -------
        None
        """
        super().__init__(model=model, output_mode=embed_mode)
        self._embed_mode = embed_mode
        self._set_forward_function()

    def _set_forward_function(self) -> None:
        self._forward_function = ViTEmbed(self._embed_mode).embed_fn

    def forward(self, x: torch.Tensor) -> Any:
        output = self._model(x)
        return self._forward_function(output)
