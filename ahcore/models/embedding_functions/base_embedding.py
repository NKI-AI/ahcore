from abc import ABC, abstractmethod
from typing import Callable

import torch


class AhCoreFeatureEmbedding(ABC):
    def __init__(self) -> None:
        self.embed_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]

    @abstractmethod
    def _set_embed_function(self) -> None:
        raise NotImplementedError
