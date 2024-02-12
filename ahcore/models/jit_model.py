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

    def start_eval(self) -> None:
        self._model.eval()

    def start_train(self) -> None:
        self._model.train()

    def place_on_cuda(self) -> None:
        self._model.cuda()

    def place_on_cpu(self) -> None:
        self._model.cpu()

    def extend_model(self, modules: dict[str, Module]) -> None:
        for key, value in modules.items():
            self._model.add_module(name=key, module=value)

    def forward(self, x: torch.Tensor) -> Any:
        return self._model(x)
