from __future__ import annotations

from enum import Enum
from functools import partial
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import torch
from dlup.data.dataset import Dataset
from pydantic import AfterValidator
from typing_extensions import Annotated


def is_positive(v: int | float) -> int | float:
    assert v > 0, f"{v} is not a positive {type(v)}"
    return v


def is_non_negative(v: int | float) -> int | float:
    assert v >= 0, f"{v} is not a non-negative {type(v)}"
    return v


PositiveInt = Annotated[int, AfterValidator(is_positive)]
PositiveFloat = Annotated[float, AfterValidator(is_positive)]
NonNegativeInt = Annotated[int, AfterValidator(is_non_negative)]
NonNegativeFloat = Annotated[float, AfterValidator(is_non_negative)]
BoundingBoxType = tuple[tuple[int, int], tuple[int, int]]
Rois = list[BoundingBoxType]
GenericNumberArray = npt.NDArray[np.int_ | np.float_]

DlupDatasetSample = dict[str, Any]
_DlupDataset = Dataset[DlupDatasetSample]


class NormalizationType(str, Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    LOGITS = "logits"

    def normalize(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self == NormalizationType.SIGMOID:
            return torch.sigmoid
        elif self == NormalizationType.SOFTMAX:
            return partial(torch.softmax, dim=0)
        elif self == NormalizationType.LOGITS:
            return lambda x: x
        else:
            raise ValueError("Function not supported")


class InferencePrecision(str, Enum):
    FP16 = "float16"
    FP32 = "float32"
    UINT8 = "uint8"

    def get_multiplier(self) -> float:
        if self == InferencePrecision.FP16:
            return 1.0
        elif self == InferencePrecision.FP32:
            return 1.0
        elif self == InferencePrecision.UINT8:
            return 255.0
        else:
            raise NotImplementedError(f"Precision {self} is not supported for {self.__class__.__name__}.")


class OutputModeBase(str, Enum):
    """Base class for embedding modes for any JIT compiled model."""

    def __str__(self) -> Any:
        return self.value


class SegmentationOutputMode(OutputModeBase):
    """
    Segmentation output modes for JIT compiled models.
    """

    DEFAULT = "default"
    # The default output mode assumes that the JIT model returns a torch tensor.
    SEGMENTATION_LOGITS = "segmentation_logits"  # Segmentation outputs without activation.
    # Extend as necessary


class ViTEmbedMode(OutputModeBase):
    """
    Embedding modes for feature extractors based on Vision Transformers.
    """

    DEFAULT = "default"
    # The default output mode assumes that the JIT model returns a torch tensor.
    CLS_ONLY = "embed_cls_only"
    PATCH_ONLY = "embed_patch_only"
    MEAN = "embed_mean"
    CONCAT_MEAN = "embed_concat_mean"
    CONCAT = "embed_concat"
    # Extend as necessary
