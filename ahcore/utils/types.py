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


class JitOutputMode(str, Enum):
    """Output modes for Ahcore foundation models.
    Since the JIT models are a general class of models that can be used in different contexts,
    the output of the model can be in different formats. This enum is used to specify the output type."""

    # Feature extraction modes from foundation models such as ViT.
    EMBED_CLS_ONLY = "embed_cls_only"
    EMBED_PATCH_ONLY = "embed_patch_only"
    EMBED_CONCAT = "embed_concat"

    # Segmentation mode
    SEGMENTATION = "segmentation"  # Segmentation outputs with activated outputs (eg: softmax is applied).
    SEGMENTATION_LOGITS = "segmentation_logits"  # Segmentation outputs without activation function.

    # Classification mode
    CLASSIFICATION = "classification"  # Classification outputs with activated outputs (eg: softmax is applied).
    CLASSIFICATION_LOGITS = "classification_logits"  # Classification outputs without activation function.

    # Detection mode
    BOUNDING_BOXES = "bounding_boxes"

    def __str__(self):
        return self.value


class FMEmbedType(str, Enum):
    """Feature map embedding types for the Ahcore foundation models."""

    CLS_TOKEN = "x_norm_clstoken"
    PATCH_TOKEN = "x_norm_patchtokens"
