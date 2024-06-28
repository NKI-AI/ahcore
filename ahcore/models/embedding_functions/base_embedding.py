from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch


class AhCoreFeatureEmbedding(ABC):
    def __init__(self) -> None:
        self.embed_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]

    @abstractmethod
    def _set_embed_function(self) -> None:
        raise NotImplementedError
