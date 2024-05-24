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


class JitModelType(str, Enum):
    """JIT model types for Ahcore."""

    FoundationModel = "foundation_model"
    SegmentationModel = "segmentation_model"

    # Extend as necessary

    def __str__(self) -> str:
        return self.value


class AhcoreFMType(str, Enum):
    """Foundation model types."""

    DINO_V2 = "Dino_v2"
    # We can extend this enum with other foundation model types as needed.


AllModelTypes = JitModelType | AhcoreFMType


class JitOutputMode(str, Enum):
    """
    Output modes for Ahcore foundation models.
    Since the JIT models are a general class of models that can be used in different contexts,
    the output of the model can be in different formats. This enum is used to specify the output mode.
    """

    # Segmentation mode
    SEGMENTATION = "segmentation"  # Segmentation outputs with activated outputs (eg: softmax is applied).

    # Extend as necessary
    def __str__(self) -> str:
        return self.value


class FMOutputMode(str, Enum):
    """Feature map output modes for the Ahcore foundation models."""

    # Feature extraction modes from foundation models such as ViT.
    EMBED_CLS_ONLY = "embed_cls_only"
    EMBED_PATCH_ONLY = "embed_patch_only"
    EMBED_CONCAT = "embed_concat"
    GENERAL_EMBEDDING = "general_embedding"  # Some foundation models do not have specific embedding outputs.

    def __str__(self) -> str:
        return self.value


AllOutputModes = JitOutputMode | FMOutputMode


class FMEmbedType(str, Enum):
    """Feature map embedding types for the Ahcore foundation models."""

    CLS_TOKEN = ("x_norm_clstoken", AhcoreFMType.DINO_V2)
    PATCH_TOKEN = ("x_norm_patchtokens", AhcoreFMType.DINO_V2)
    # Add other FMEmbedType values here if needed, specifying the corresponding FMType

    def __new__(cls, value: str, fm_type: AhcoreFMType) -> "FMEmbedType":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.fm_type = fm_type
        return obj

    @property
    def fm_type(self) -> AhcoreFMType:
        return self._fm_type

    @fm_type.setter
    def fm_type(self, fm_type: AhcoreFMType) -> None:
        self._fm_type = fm_type

    @classmethod
    def for_type(cls, fm_type: AllModelTypes) -> list[str]:
        return [member for member in cls if member.fm_type == fm_type]
