from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch


class AhCoreFeatureEmbedding(ABC):
    def __init__(self, embedding_mode: str):
        self._embedding_mode = embedding_mode
        self.embed_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]

    @property
    def embedding_mode(self) -> Any:
        return self._embedding_mode

    @abstractmethod
    def _set_embed_function(self) -> None:
        raise NotImplementedError
