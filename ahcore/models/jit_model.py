from pathlib import Path
from typing import Any

import torch
from torch.jit import ScriptModule, load
from torch.nn import Module

from ahcore.utils.types import FMEmbedType, JitOutputMode


class AhcoreJitModel(ScriptModule):
    def __init__(self, model: ScriptModule, output_mode: str) -> None:
        super().__init__()  # type: ignore
        self._model = model
        if output_mode == JitOutputMode.EMBED_CLS_ONLY:
            output_function = self.cls_only
        elif output_mode == JitOutputMode.EMBED_PATCH_ONLY:
            output_function = self.patch_only
        elif output_mode == JitOutputMode.EMBED_CONCAT:
            output_function = self.concat_embedding
        elif output_mode == JitOutputMode.SEGMENTATION:
            output_function = self.segmentation_map
        else:
            raise ValueError(f"Unsupported output mode: {self.output_mode}")

        self._forward_function = output_function

    @classmethod
    def from_jit_path(cls, jit_path: Path, output_mode: str) -> Any:
        model = load(jit_path)  # type: ignore
        return cls(model, output_mode)

    def segmentation_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def cls_only(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)[FMEmbedType.CLS_TOKEN]

    def patch_only(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)[FMEmbedType.PATCH_TOKEN]

    def concat_embedding(self, x: torch.Tensor) -> torch.Tensor:
        output = self._model(x)
        cls_token = output[FMEmbedType.CLS_TOKEN].unsqueeze(1)
        mean_patch_token = output[FMEmbedType.PATCH_TOKEN].mean(dim=1).unsqueeze(1)
        tokens = torch.cat([cls_token, mean_patch_token], dim=1)
        output = tokens.mean(dim=1)
        return output

    def extend_model(self, modules: dict[str, Module]) -> None:
        for key, value in modules.items():
            self._model.add_module(name=key, module=value)

    def forward(self, x: torch.Tensor) -> Any:
        return self._forward_function(x)
