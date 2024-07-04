from typing import Any

import torch
from torch.jit import ScriptModule

from ahcore.models.base_jit_model import BaseAhcoreJitModel
from ahcore.utils.types import SegmentationOutputMode


class SegmentationJitModel(BaseAhcoreJitModel):
    """
    This class is a wrapper for the segmentation models in Ahcore.
    It provides a general interface for the jit compiled segmentation models.
    """

    def __init__(self, model: ScriptModule, output_mode: str) -> None:
        """
        Constructor for the SegmentationModel class.

        Parameters
        ----------
        model : ScriptModule
            The jit compiled segmentation model.

        output_mode: OutputModeBase
            The output mode of the model. This is used to determine the forward function of the model.
        """
        self._output_mode = SegmentationOutputMode(output_mode)
        super().__init__(model=model)
        self._set_forward_function()

    def _set_forward_function(self) -> None:
        if self._output_mode == SegmentationOutputMode.ACTIVATED_OUTPUTS:
            self._forward_function = self._segmentation_with_activation
        else:
            raise NotImplementedError(f"Output mode {self._output_mode} is not supported for {self.__class__.__name__}")

    def _segmentation_with_activation(self, x: torch.Tensor) -> Any:
        return self._model(x)

    def forward(self, x: torch.Tensor) -> Any:
        return self._forward_function(x)
