from pathlib import Path
from typing import Any, Callable

import torch
from torch.jit import ScriptModule, load
from torch.nn import Module

from ahcore.utils.types import OutputModeBase, SegmentationOutputMode


class AhcoreJitModel(ScriptModule):
    """
    Base class for the jit compiled models in Ahcore.
    """

    def __init__(self, model: ScriptModule, output_mode: OutputModeBase) -> None:
        """
        Constructor for the AhcoreJitModel class.

        Parameters
        ----------
        model: ScriptModule
            The jit compiled model.

        output_mode: OutputModeBase
            The output mode of the model. This is used to determine the forward function of the model.

        Returns
        -------
        None
        """
        super().__init__()  # type: ignore
        self._model = model
        self._output_mode = output_mode
        self._forward_function: Callable[[Any], torch.Tensor]

    @classmethod
    def from_jit_path(cls, jit_path: Path, output_mode: OutputModeBase) -> Any:
        """
        Load a jit compiled model from a file path.

        Parameters
        ----------
        jit_path : Path
            The path to the jit compiled model.

        output_mode : OutputModeBase
            The output mode of the model. This is used to determine the forward function of the model.

        Returns
        -------
        An instance of the AhcoreJitModel class.
        """
        model = load(jit_path)  # type: ignore
        return cls(model, output_mode)

    def _set_forward_function(self) -> None:
        """
        Set the forward function of the model.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def extend_model(self, modules: dict[str, Module]) -> None:
        """
        Add modules to a jit compiled model.

        Parameters
        ----------
        modules : dict[str, Module]
            A dictionary of modules to add to the model.

        Returns
        -------
        None
        """
        for key, value in modules.items():
            self._model.add_module(name=key, module=value)

    def forward(self, x: torch.Tensor) -> Any:
        raise NotImplementedError


class SegmentationModel(AhcoreJitModel):
    """
    This class is a wrapper for the segmentation models in Ahcore.
    It provides a general interface for the jit compiled segmentation models.
    """

    def __init__(self, model: ScriptModule, output_mode: SegmentationOutputMode) -> None:
        """
        Constructor for the SegmentationModel class.

        Parameters
        ----------
        model : ScriptModule
            The jit compiled segmentation model.

        output_mode: OutputModeBase
            The output mode of the model. This is used to determine the forward function of the model.
        """
        super().__init__(model=model, output_mode=output_mode)
        self._set_forward_function()

    def _set_forward_function(self) -> None:
        if self._output_mode == SegmentationOutputMode.SEGMENTATION:
            self._forward_function = self._segmentation
        else:
            raise NotImplementedError(f"Output mode {self._output_mode} is not supported for {self.__class__.__name__}")

    def _segmentation(self, x: torch.Tensor) -> Any:
        return self._model(x)

    def forward(self, x: torch.Tensor) -> Any:
        return self._forward_function(x)
