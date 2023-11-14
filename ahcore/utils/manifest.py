"""
All utilities to parse manifests into datasets. A manifest is a database containing the description of a dataset.
See the documentation for more information and examples.
"""

from __future__ import annotations

import functools
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Generator, Literal, Optional, Type, TypedDict, cast

from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import RegionFromWsiDatasetSample, TiledWsiDataset, TileSample
from dlup.experimental_backends import ImageBackend  # type: ignore
from dlup.tiling import GridOrder, TilingMode
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ahcore.exceptions import RecordNotFoundError
from ahcore.utils.data import DataDescription
from ahcore.utils.database_models import (
    Base,
    CategoryEnum,
    Image,
    ImageAnnotations,
    Manifest,
    Mask,
    Patient,
    Split,
    SplitDefinitions,
)
from ahcore.utils.io import get_enum_key_from_value, get_logger
from ahcore.utils.rois import compute_rois
from ahcore.utils.types import PositiveFloat, PositiveInt, Rois

logger = get_logger(__name__)

_AnnotationReturnTypes = WsiAnnotations | SlideImage


class _AnnotationReadersDict(TypedDict):
    ASAP_XML: Callable[[Path], WsiAnnotations]
    GEOJSON: Callable[[Path], WsiAnnotations]
    PYVIPS: Callable[[Path], SlideImage]
    TIFFFILE: Callable[[Path], SlideImage]
    OPENSLIDE: Callable[[Path], SlideImage]


_AnnotationReaders: _AnnotationReadersDict = {
    "ASAP_XML": WsiAnnotations.from_asap_xml,
    "GEOJSON": WsiAnnotations.from_geojson,
    "PYVIPS": functools.partial(SlideImage.from_file_path, backend=ImageBackend.PYVIPS),
    "TIFFFILE": functools.partial(SlideImage.from_file_path, backend=ImageBackend.TIFFFILE),
    "OPENSLIDE": functools.partial(SlideImage.from_file_path, backend=ImageBackend.OPENSLIDE),
}


def parse_annotations_from_record(
    annotations_root: Path, record: list[Mask] | list[ImageAnnotations]
) -> _AnnotationReturnTypes | None:
    """
    Parse the annotations from a record of type ImageAnnotations.

    Parameters
    ----------
    annotations_root : Path
        The root directory of the annotations.
    record : list[Type[ImageAnnotations]]
        The record containing the annotations.

    Returns
    -------
    WsiAnnotations
        The parsed annotations.
    """
    if not record:
        return None
    assert len(record) == 1

    valid_readers = list(_AnnotationReaders.keys())
    reader_name = cast(
        Literal["ASAP_XML", "GEOJSON", "PYVIPS", "TIFFFILE", "OPENSLIDE"],
        record[0].reader,
    )

    if reader_name not in valid_readers:
        raise ValueError(f"Invalid reader: {record[0].reader}")
    assert reader_name in valid_readers

    filename = record[0].filename

    try:
        reader_func = _AnnotationReaders[reader_name]
    except KeyError:
        raise NotImplementedError(f"Reader {reader_name} not implemented.")

    return reader_func(annotations_root / filename)


def get_mask_and_annotations_from_record(
    annotations_root: Path, record: Image
) -> tuple[_AnnotationReturnTypes | None, _AnnotationReturnTypes | None]:
    """
    Get the mask and annotations from a record of type Image.

    Parameters
    ----------
    annotations_root : Path
        The root directory of the annotations.
    record : Type[Image]
        The record containing the mask and annotations.

    Returns
    -------
    tuple[WsiAnnotations, WsiAnnotations]
        The mask and annotations.
    """
    _masks = parse_annotations_from_record(annotations_root, record.masks)
    _annotations = parse_annotations_from_record(annotations_root, record.annotations)
    return _masks, _annotations


def _get_rois(mask: WsiAnnotations | None, data_description: DataDescription, stage: str) -> Optional[Rois]:
    if (mask is None) or (stage != "fit") or (not data_description.convert_mask_to_rois):
        return None

    tile_size = data_description.training_grid.tile_size
    tile_overlap = data_description.training_grid.tile_overlap

    return compute_rois(mask, tile_size=tile_size, tile_overlap=tile_overlap, centered=True)


class DataManager:
    def __init__(self, database_uri: str) -> None:
        self._database_uri = database_uri
        self.__session: Optional[Session] = None
        self._logger = get_logger(type(self).__name__)

    @property
    def _session(self) -> Session:
        if self.__session is None:
            self.__session = open_db(self._database_uri)
        return self.__session

    @staticmethod
    def _ensure_record(record: Any, description: str) -> None:
        """Raises an error if the record is None."""
        if not record:
            raise RecordNotFoundError(f"{description} not found.")

    def get_records_by_split(
        self,
        manifest_name: str,
        split_version: str,
        split_category: Optional[str] = None,
    ) -> Generator[Patient, None, None]:
        manifest = self._session.query(Manifest).filter_by(name=manifest_name).first()
        self._ensure_record(manifest, f"Manifest with name {manifest_name}")

        split_definition = self._session.query(SplitDefinitions).filter_by(version=split_version).first()
        self._ensure_record(split_definition, f"Split definition with version {split_version}")

        # This is because mypy is complaining otherwise,
        # but _ensure_record effectively ensures that the record is not None
        assert manifest is not None
        assert split_definition is not None
        query = (
            self._session.query(Patient)
            .join(Split)
            .filter(
                Patient.manifest_id == manifest.id,
                Split.split_definition_id == split_definition.id,
            )
        )

        if split_category is not None:
            split_category_key = get_enum_key_from_value(split_category, CategoryEnum)
            query = query.filter(Split.category == split_category_key)

        patients = query.all()

        self._logger.info(
            f"Found {len(patients)} patients for split {split_category if split_category else 'all categories'}"
        )
        for patient in patients:
            yield patient

    def get_image_metadata_by_split(
        self,
        manifest_name: str,
        split_version: str,
        split_category: Optional[str] = None,
    ) -> Generator[ImageMetadata, None, None]:
        """
        Yields the metadata of images for a given manifest name, split version, and optional split category.

        Parameters
        ----------
        manifest_name : str
            The name of the manifest.
        split_version : str
            The version of the split.
        split_category : Optional[str], default=None
            The category of the split (e.g., "fit", "validate", "test").

        Yields
        -------
        ImageMetadata
            The metadata of the image.
        """
        for patient in self.get_records_by_split(manifest_name, split_version, split_category):
            for image in patient.images:
                yield fetch_image_metadata(image)

    def get_image_metadata_by_patient(self, patient_code: str) -> list[ImageMetadata]:
        """
        Fetch the metadata for the images associated with a specific patient.

        Parameters
        ----------
        patient_code : str
            The unique code of the patient.

        Returns
        -------
        list[ImageData]
            A list of metadata for all images associated with the patient.
        """
        patient = self._session.query(Patient).filter_by(patient_code=patient_code).first()
        self._ensure_record(patient, f"Patient with code {patient_code} not found")
        assert patient is not None  # for mypy
        return [fetch_image_metadata(image) for image in patient.images]

    def get_image_by_filename(self, filename: str) -> Image:
        """
        Fetch the metadata for an image based on its filename.

        Parameters
        ----------
        filename : str
            The filename of the image.

        Returns
        -------
        Image
            The image from the database.
        """
        image = self._session.query(Image).filter_by(filename=filename).first()
        self._ensure_record(image, f"Image with filename {filename} not found")
        assert image
        return image

    def get_image_metadata_by_id(self, image_id: int) -> ImageMetadata:
        """
        Fetch the metadata for an image based on its ID.

        Parameters
        ----------
        image_id : int
            The ID of the image.

        Returns
        -------
        ImageMetadata
            Metadata of the image.
        """
        image = self._session.query(Image).filter_by(id=image_id).first()
        self._ensure_record(image, f"No image found with ID {image_id}")
        assert image is not None  # mypy
        return fetch_image_metadata(image)

    def __enter__(self) -> "DataManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Literal[False]:
        if self._session is not None:
            self.close()
        return False

    def close(self) -> None:
        if self.__session is not None:
            self.__session.close()
            self.__session = None


def datasets_from_data_description(
    db_manager: DataManager,
    data_description: DataDescription,
    transform: Callable[[TileSample], RegionFromWsiDatasetSample] | None,
    stage: str,
) -> Generator[TiledWsiDataset, None, None]:
    logger.info(f"Reading manifest from {data_description.manifest_database_uri} for stage {stage}")

    image_root = data_description.data_dir
    annotations_root = data_description.annotations_dir

    assert isinstance(stage, str), "Stage should be a string."

    if stage == "fit":
        grid_description = data_description.training_grid
    else:
        grid_description = data_description.inference_grid

    records = db_manager.get_records_by_split(
        manifest_name=data_description.manifest_name,
        split_version=data_description.split_version,
        split_category=stage,
    )
    for record in records:
        labels = [(str(label.key), str(label.value)) for label in record.labels] if record.labels else None

        for image in record.images:
            mask, annotations = get_mask_and_annotations_from_record(annotations_root, image)
            assert isinstance(mask, WsiAnnotations) or (mask is None)
            rois = _get_rois(mask, data_description, stage)
            mask_threshold = 0.0 if stage != "fit" else data_description.mask_threshold

            dataset = TiledWsiDataset.from_standard_tiling(
                path=image_root / image.filename,
                mpp=grid_description.mpp,
                tile_size=grid_description.tile_size,
                tile_overlap=grid_description.tile_overlap,
                tile_mode=TilingMode.overflow,
                grid_order=GridOrder.C,
                crop=False,
                mask=mask,
                mask_threshold=mask_threshold,
                rois=rois if rois is not None else None,
                annotations=annotations if stage != "predict" else None,
                labels=labels,  # type: ignore
                transform=transform,
                backend=ImageBackend[str(image.reader)],
                overwrite_mpp=(image.mpp, image.mpp),
                limit_bounds=False if rois is not None else True,
            )

            yield dataset


class ImageMetadata(BaseModel):
    """Model to hold image metadata"""

    class Config:
        frozen = True

    filename: Path
    height: PositiveInt
    width: PositiveInt
    mpp: PositiveFloat


def open_db(database_uri: str) -> Session:
    """Open a database connection.

    Parameters
    ----------
    database_uri : str
        The URI of the database.

    Returns
    -------
    Session
        The database session.
    """
    engine = create_engine(database_uri)
    create_tables(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def create_tables(engine: Engine) -> None:
    """Create the database tables."""
    Base.metadata.create_all(bind=engine)


def fetch_image_metadata(image: Image) -> ImageMetadata:
    """Extract metadata from an Image object."""
    return ImageMetadata(
        filename=Path(image.filename),
        height=int(image.height),
        width=int(image.width),
        mpp=float(image.mpp),
    )
