"""Utilities to describe the dataset to be used and the way it should be parsed."""

from __future__ import annotations

import hashlib
import pickle
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import BaseModel
from sqlalchemy import create_engine, exists
from sqlalchemy.engine import Engine
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session, sessionmaker

from ahcore.utils.database_models import Base, Manifest, OnTheFlyBase
from ahcore.utils.types import NonNegativeInt, PositiveFloat, PositiveInt


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


def open_db_from_engine(engine: Engine) -> Session:
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def open_db_from_uri(
    uri: str,
    ensure_exists: bool = True,
) -> Session:
    """Open a database connection from a uri"""

    # Set up the engine if no engine is given and uri is given.
    engine = create_engine(uri)

    if not ensure_exists:
        # Create tables if they don't exist
        create_tables(engine, base=Base)
    else:
        # Check if the "manifest" table exists
        inspector = inspect(engine)
        if "manifest" not in inspector.get_table_names():
            raise RuntimeError("Manifest table does not exist. Likely you have set the wrong URI.")

        # Check if the "manifest" table is not empty
        with engine.connect() as connection:
            result = connection.execute(exists().where(Manifest.id.isnot(None)).select())
            if not result.scalar():
                raise RuntimeError("Manifest table is empty. Likely you have set the wrong URI.")

    return open_db_from_engine(engine)


def create_tables(engine: Engine, base: type[Base] | type[OnTheFlyBase]) -> None:
    """Create the database tables."""
    base.metadata.create_all(bind=engine)


class GridDescription(BaseModel):
    mpp: Optional[PositiveFloat]
    tile_size: Tuple[PositiveInt, PositiveInt]
    tile_overlap: Tuple[NonNegativeInt, NonNegativeInt]
    output_tile_size: Optional[Tuple[int, int]] = None


class DataDescription(BaseModel):
    mask_label: Optional[str] = None
    mask_threshold: Optional[float] = None  # This is only used for training
    roi_name: Optional[str] = None
    num_classes: NonNegativeInt
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
    apply_color_profile: bool = False


class OnTheFlyDataDescription(BaseModel):
    # Required
    data_dir: Path
    glob_pattern: str
    num_classes: NonNegativeInt
    inference_grid: GridDescription

    # Preset?
    convert_mask_to_rois: bool = True
    use_roi: bool = True
    apply_color_profile: bool = False

    # Explicitly optional
    annotations_dir: Optional[Path] = None  # May be used to provde a mask.
    mask_label: Optional[str] = None
    mask_threshold: Optional[float] = None  # This is only used for training
    roi_name: Optional[str] = None
    index_map: Optional[Dict[str, int]]
    remap_labels: Optional[Dict[str, str]] = None
