from typing import Any

import torch
from torch.jit import ScriptModule

from ahcore.models.embedding_functions.vit_embed import ViTEmbed
from ahcore.models.jit_model import BaseAhcoreJitModel
from ahcore.utils.types import OutputModeBase, ViTEmbedMode


class DinoV2JitModel(BaseAhcoreJitModel):
    """
    This class is a wrapper for the DinoV2 foundation model in Ahcore.
    """

    def __init__(self, model: ScriptModule, output_mode: OutputModeBase) -> None:
        """
        Constructor for the DinoV2 class.
        This can be a feature extractor based on very large pre-trained model.

        Parameters
        ----------
        model : ScriptModule
            The jit compiled model.

        output_mode : OutputModeBase
            The output mode of the model. This is used to determine the forward function of the model.

        Returns
        -------
        None
        """
        super().__init__(model=model, output_mode=output_mode)
        self._set_forward_function()

    def _set_forward_function(self) -> None:
        if self._output_mode == ViTEmbedMode.DEFAULT:  # Assume that the JIT model returns a tensor
            self._forward_function = lambda x: x
        else:
            self._forward_function = ViTEmbed(self._output_mode).embed_fn  # More complex forward function

    def forward(self, x: torch.Tensor) -> Any:
        output = self._model(x)
        return self._forward_function(output)
