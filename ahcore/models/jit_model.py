from pathlib import Path
from typing import Any

import torch
from torch.jit import ScriptModule, load
from torch.nn import Module


class AhcoreJitModel(ScriptModule):
    def __init__(self, model: ScriptModule) -> None:
        super().__init__()  # type: ignore
        self._model = model

    @classmethod
    def from_jit_path(cls, jit_path: Path) -> Any:
        model = load(jit_path)  # type: ignore
        return cls(model)

    def extend_model(self, modules: dict[str, Module]) -> None:
        for key, value in modules.items():
            self._model.add_module(name=key, module=value)

    def forward(self, x: torch.Tensor) -> Any:
        return self._model(x)
