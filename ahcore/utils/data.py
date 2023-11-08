"""Utilities to describe the dataset to be used and the way it should be parsed."""
from __future__ import annotations

import hashlib
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from ahcore.utils.types import DlupDatasetSample, NonNegativeInt, PositiveFloat, PositiveInt


def basemodel_to_uuid(base_model: BaseModel) -> uuid.UUID:
    """Create a unique identifier for a pydantic model.

    This is done by pickling the object, and computing the sha256 hash of the pickled object and converting this to
    an UUID. The UUID is generated using the sha256 hash as a namespace, ensuring similar lengths. The chance of
    a collision is astronomically small.

    Arguments
    ---------
    base_model: BaseModel
        The BaseModel to create a unique identifier for.

    Returns
    -------
    uuid.UUID
        A unique identifier for the BaseModel.
    """
    # Serialize the object
    serialized_data = pickle.dumps(base_model.model_dump())

    # Generate a sha256 hash of the serialized data
    obj_hash = hashlib.sha256(serialized_data).digest()

    # Use the hash as a namespace to generate a UUID
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, obj_hash.hex())

    return unique_id


def collate_fn_annotations(batch: list[DlupDatasetSample]) -> Any:
    def _collate_fn_ann_type(
        batch: list[DlupDatasetSample], ann_type: Literal["points", "boxes"]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        # any function short circuits at first True evaluation and pre_transform places at least empty tensor
        if any(sample.get(ann_type) is not None for sample in batch):
            _ann_coords = [sample.pop(ann_type) for sample in batch]
            _ann_labels = [sample.pop(f"{ann_type}_labels") for sample in batch]
            return (
                pad_sequence(_ann_coords, batch_first=True, padding_value=torch.nan).float(),
                pad_sequence(_ann_labels, batch_first=True, padding_value=-1),
            )
        return None

    _padded_points = _collate_fn_ann_type(batch, "points")
    _padded_boxes = _collate_fn_ann_type(batch, "boxes")
    collated_batch = default_collate(batch)
    if _padded_points is not None:
        collated_batch["points"] = _padded_points[0]
        collated_batch["points_labels"] = _padded_points[1]
    if _padded_boxes is not None:
        collated_batch["boxes"] = _padded_boxes[0]
        collated_batch["boxes_labels"] = _padded_boxes[1]
    return collated_batch


class GridDescription(BaseModel):
    mpp: Optional[PositiveFloat]
    tile_size: Tuple[PositiveInt, PositiveInt]
    tile_overlap: Tuple[NonNegativeInt, NonNegativeInt]
    output_tile_size: Optional[Tuple[int, int]] = None


class DataDescription(BaseModel):
    mask_label: Optional[str] = None
    mask_threshold: Optional[float] = None  # This is only used for training
    roi_name: Optional[str] = None
    num_classes: PositiveInt
    data_dir: Path
    manifest_database_uri: str
    manifest_name: str
    split_version: str
    annotations_dir: Path
    training_grid: GridDescription
    inference_grid: GridDescription
    index_map: Optional[Dict[str, int]]
    remap_labels: Optional[Dict[str, str]] = None
    use_class_weights: Optional[bool] = False
    convert_mask_to_rois: bool = True
    use_roi: bool = True
    apply_color_profile: bool = True

    use_points: bool = False
    use_boxes: bool = False
    point_radius_microns: Optional[float] = None
