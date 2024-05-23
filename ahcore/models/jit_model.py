from pathlib import Path
from typing import Any, Callable

import torch
from torch.jit import ScriptModule, load
from torch.nn import Module

from ahcore.utils.types import AllModelTypes, AllOutputModes, FMEmbedType, FMOutputMode, JitModelType, JitOutputMode


class AhcoreJitModel(ScriptModule):
    def __init__(self, model: ScriptModule, model_type: AllModelTypes, output_mode: AllOutputModes) -> None:
        super().__init__()  # type: ignore
        self._model = model
        self._model_type = model_type
        self._output_mode = output_mode
        self._forward_function: Callable[[torch.Tensor], torch.Tensor]

    @classmethod
    def from_jit_path(cls, jit_path: Path, model_type: AllModelTypes, output_mode: AllOutputModes) -> Any:
        model = load(jit_path)  # type: ignore
        return cls(model, model_type, output_mode)

    def extend_model(self, modules: dict[str, Module]) -> None:
        for key, value in modules.items():
            self._model.add_module(name=key, module=value)

    def forward(self, x: torch.Tensor) -> Any:
        return self._forward_function(x)


class FoundationalModel(AhcoreJitModel):
    """
    This class is a wrapper for the foundation models in Ahcore.
    It provides a general interface for the jit compiled foundation models that can be used in different contexts.
    """

    def __init__(self, model: ScriptModule, model_type: AllModelTypes, output_mode: AllOutputModes) -> None:
        super().__init__(model=model, model_type=JitModelType.FoundationModel, output_mode=output_mode)
        self._fm_type = model_type
        self._fm_embed_type = FMEmbedType.for_type(self._fm_type)
        if self._output_mode == FMOutputMode.EMBED_CLS_ONLY:
            self._forward_function = self._cls_only
        elif self._output_mode == FMOutputMode.EMBED_PATCH_ONLY:
            self._forward_function = self._patch_only
        elif self._output_mode == FMOutputMode.EMBED_CONCAT:
            self._forward_function = self._concat_embedding
        elif self._output_mode == FMOutputMode.GENERAL_EMBEDDING:
            self._forward_function = self._general_embedding
        else:
            raise NotImplementedError(f"Output mode {self._output_mode} is not supported for {self.__class__.__name__}")

    @classmethod
    def from_jit_path(cls, jit_path: Path, model_type: AllModelTypes, output_mode: AllOutputModes) -> Any:
        model = load(jit_path)  # type: ignore
        return cls(model, model_type, output_mode)

    def _cls_only(self, x: torch.Tensor) -> Any:
        return self._model(x)[FMEmbedType.CLS_TOKEN.value]

    def _patch_only(self, x: torch.Tensor) -> Any:
        return self._model(x)[FMEmbedType.PATCH_TOKEN.value]

    def _concat_embedding(self, x: torch.Tensor) -> Any:
        output = self._model(x)
        cls_token = output[FMEmbedType.CLS_TOKEN.value].unsqueeze(1)
        mean_patch_token = output[FMEmbedType.PATCH_TOKEN.value].mean(dim=1).unsqueeze(1)
        tokens = torch.cat([cls_token, mean_patch_token], dim=1)
        output = tokens.mean(dim=1)
        return output

    def _general_embedding(self, x: torch.Tensor) -> Any:
        # This function is for the foundation models that do not have specific embedding outputs.
        return self._model(x)

    def forward(self, x: torch.Tensor) -> Any:
        return self._forward_function(x)


class SegmentationModel(AhcoreJitModel):
    """
    This class is a wrapper for the segmentation models in Ahcore.
    It provides a general interface for the jit compiled segmentation models that can be used in different contexts.
    """

    def __init__(self, model: ScriptModule, model_type: AllModelTypes, output_mode: AllOutputModes) -> None:
        super().__init__(model=model, model_type=JitModelType.SegmentationModel, output_mode=output_mode)
        if self._output_mode == JitOutputMode.SEGMENTATION:
            self._forward_function = self._segmentation
        else:
            raise NotImplementedError(f"Output mode {self._output_mode} is not supported for {self.__class__.__name__}")

    @classmethod
    def from_jit_path(cls, jit_path: Path, model_type: AllModelTypes, output_mode: AllOutputModes) -> Any:
        model = load(jit_path)  # type: ignore
        return cls(model, model_type, output_mode)

    def _segmentation(self, x: torch.Tensor) -> Any:
        return self._model(x)

    def forward(self, x: torch.Tensor) -> Any:
        return self._forward_function(x)
