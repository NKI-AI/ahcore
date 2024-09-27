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
    SIGMOID = "SIGMOID"
    SOFTMAX = "SOFTMAX"
    LOGITS = "LOGITS"

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


class SegmentationOutputMode(str, Enum):
    """
    Segmentation output modes for JIT compiled models.
    """

    ACTIVATED_OUTPUTS = "activated_outputs"  # Segmentation outputs with activation.
    # Extend as necessary


class ViTEmbedMode(str, Enum):
    """
    Embedding modes for feature extractors based on Vision Transformers.
    """

    CLS_ONLY = "embed_cls_only"
    PATCH_ONLY = "embed_patch_only"
    MEAN = "embed_mean"
    CONCAT_MEAN = "embed_concat_mean"
    CONCAT = "embed_concat"
    # Extend as necessary


class ScannerEnum(Enum):
    SVS = ("svs", "Aperio")
    MRXS = ("mrxs", "P1000")
    DEFAULT = ("default", "Unknown Scanner")

    def __init__(self, extension, scanner_name):
        self.extension = extension
        self.scanner_name = scanner_name

    @classmethod
    def get_scanner_name(cls, file_extension):
        for scanner in cls:
            if scanner.extension == file_extension:
                return scanner.scanner_name
        # Return a default value if extension is not found
        return cls.DEFAULT.scanner_name


class LoggerEnum(Enum):
    TENSORBOARD = "tensorboard"
    MLFLOW = "mlflow"
    UNKNOWN = "unknown"
    # Extend as necessary
