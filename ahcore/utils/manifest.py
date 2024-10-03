"""
All utilities to parse manifests into datasets. A manifest is a database containing the description of a dataset.
See the documentation for more information and examples.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Generator, Literal, Optional, Tuple, Type, TypedDict, cast

from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.backends import ImageBackend as DLUPImageBackend
from dlup.data.dataset import RegionFromWsiDatasetSample, TiledWsiDataset, TileSample
from dlup.tiling import GridOrder, TilingMode
from pydantic import BaseModel
from sqlalchemy import Column, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import exists

from ahcore.backends import ImageBackend
from ahcore.exceptions import RecordNotFoundError
from ahcore.utils.data import DataDescription
from ahcore.utils.database_models import (
    Base,
    CategoryEnum,
    FeatureDescription,
    Image,
    ImageAnnotations,
    ImageFeature,
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
    DARWIN_JSON: Callable[[Path], WsiAnnotations]
    GEOJSON: Callable[[Path], WsiAnnotations]
    PYVIPS: Callable[[Path], SlideImage]
    TIFFFILE: Callable[[Path], SlideImage]
    OPENSLIDE: Callable[[Path], SlideImage]


_AnnotationReaders: _AnnotationReadersDict = {
    "ASAP_XML": WsiAnnotations.from_asap_xml,
    "DARWIN_JSON": WsiAnnotations.from_darwin_json,
    "GEOJSON": WsiAnnotations.from_geojson,
    "PYVIPS": functools.partial(SlideImage.from_file_path, backend=DLUPImageBackend.PYVIPS),
    "TIFFFILE": functools.partial(
        SlideImage.from_file_path,
        backend=DLUPImageBackend.TIFFFILE,
    ),
    "OPENSLIDE": functools.partial(SlideImage.from_file_path, backend=DLUPImageBackend.OPENSLIDE),
}


class ImageInfoDict(TypedDict):
    image_path: Optional[Path]
    tile_size: Optional[Tuple[int, int]]
    tile_overlap: Optional[Tuple[int, int]]
    backend: Optional[ImageBackend]
    mpp: Optional[float]
    overwrite_mpp: Optional[float]
    tile_mode: Optional[TilingMode]
    output_tile_size: Optional[Tuple[int, int]]
    mask: Optional[_AnnotationReturnTypes]
    mask_threshold: Optional[float]
    rois: Optional[Rois]
    annotations: Optional[_AnnotationReturnTypes]


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


def get_labels_from_record(record: Image | Patient) -> list[tuple[str, str]] | None:
    """Get the labels from a record of type Image or Patient.

    Parameters
    ----------
    record : Image | Patient
        The record containing the labels.

    Returns
    -------
    list[tuple[str, str]] | None
        The labels if they exists, else None.
    """
    _labels = [(str(label.key), str(label.value)) for label in record.labels] if record.labels else None
    return _labels


def get_relevant_feature_info_from_record(
    record: ImageFeature, data_description: DataDescription, feature_description: FeatureDescription
) -> ImageInfoDict:
    """Get the features from a record of type Image.

    Parameters
    ----------
    record : Type[Image]
        The record containing the features.

    Returns
    -------
    tuple[Path, PositiveFloat, tuple[PositiveInt, PositiveInt],
    tuple[PositiveInt, PositiveInt], TilingMode, ImageBackend, PositiveFloat]
        The features of the image.
    """
    image_path = data_description.data_dir / record.filename
    mpp = float(feature_description.mpp)
    tile_size = (
        int(record.num_tiles),
        1,
    )  # this would load all the features in one go --> can be extended to only load relevant tile level features
    tile_overlap = (0, 0)

    backend = ImageBackend[str(record.reader)].value
    overwrite_mpp = float(feature_description.mpp)

    output_dict: ImageInfoDict = {
        "image_path": image_path,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "backend": backend,
        "mpp": mpp,
        "overwrite_mpp": overwrite_mpp,
        "tile_mode": TilingMode.skip,
        "output_tile_size": None,
        "mask": None,
        "mask_threshold": None,
        "rois": None,
        "annotations": None,
    }

    return output_dict


def get_relevant_image_info_from_record(
    image: Image, data_description: DataDescription, annotations_root: Path, stage: str
) -> ImageInfoDict:
    if stage == "fit":
        grid_description = data_description.training_grid
    else:
        grid_description = data_description.inference_grid

    if grid_description is None:
        raise ValueError(f"Grid (for stage {stage}) is not defined in the data description.")

    mask, annotations = get_mask_and_annotations_from_record(annotations_root, image)
    assert isinstance(mask, WsiAnnotations) or (mask is None) or isinstance(mask, SlideImage)
    mask_threshold = 0.0 if stage != "fit" else data_description.mask_threshold
    rois = _get_rois(mask, data_description, stage)

    image_path = data_description.data_dir / image.filename
    tile_size = grid_description.tile_size
    tile_overlap = grid_description.tile_overlap
    backend = DLUPImageBackend[str(image.reader)]
    mpp = getattr(grid_description, "mpp", 1.0)
    overwrite_mpp = float(image.mpp)
    tile_mode = (
        TilingMode(data_description.tiling_mode) if data_description.tiling_mode is not None else TilingMode.overflow
    )
    output_tile_size = getattr(grid_description, "output_tile_size", None)

    output_dict: ImageInfoDict = {
        "image_path": image_path,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "backend": backend,
        "mpp": mpp,
        "overwrite_mpp": overwrite_mpp,
        "tile_mode": tile_mode,
        "output_tile_size": output_tile_size,
        "mask": mask,
        "mask_threshold": mask_threshold,
        "rois": rois,
        "annotations": annotations,
    }

    return output_dict


def _get_rois(mask: Optional[_AnnotationReturnTypes], data_description: DataDescription, stage: str) -> Optional[Rois]:
    if (mask is None) or (stage != "fit") or (not data_description.convert_mask_to_rois):
        return None

    assert data_description.training_grid is not None
    assert isinstance(mask, WsiAnnotations)  # this is necessary for the compute_rois to work

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
        try:
            self._ensure_record(manifest, f"Manifest with name {manifest_name}")
        except RecordNotFoundError as e:
            raise RecordNotFoundError(
                f"Manifest with name {manifest_name} not found. "
                f"Available manifest names: {', '.join([str(m.name) for m in self._session.query(Manifest).all()])}"
            ) from e

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

    def get_image_features_by_image_and_feature_version(
        self, image_id: Column[int], feature_version: str | None
    ) -> Tuple[ImageFeature | None, FeatureDescription]:
        """
        Fetch the features for an image based on its ID and feature version.

        Parameters
        ----------
        image_id : int
            The ID of the image.
        feature_version : str
            The version of the features.

        Returns
        -------
        ImageFeature
            The features of the image.
        """
        if feature_version is None:
            raise ValueError("feature_version cannot be None")

        feature_description = self._session.query(FeatureDescription).filter_by(version=feature_version).first()

        if feature_description is None:
            raise ValueError(f"Couldn't find feature description matching version {feature_version}")

        image_feature = (
            self._session.query(ImageFeature)
            .filter_by(image_id=image_id, feature_description_id=feature_description.id)
            .first()
        )
        if not image_feature:
            logging.warning(f"No features found for image ID {image_id} and feature version {feature_version}")
        # todo: make sure that this only allows to run one ImageFeature,
        #  I think it should be good bc of the unique constraint
        return image_feature, feature_description

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


def get_image_info(
    db_manager: DataManager, data_description: DataDescription, image: Image, stage: str
) -> ImageInfoDict:
    # Initialize the output dictionary with all keys set to None
    image_info: ImageInfoDict = {
        "image_path": None,
        "tile_size": None,
        "tile_overlap": None,
        "backend": None,
        "mpp": None,
        "overwrite_mpp": None,
        "tile_mode": None,
        "output_tile_size": None,
        "mask": None,
        "mask_threshold": None,
        "rois": None,
        "annotations": None,
    }

    annotations_root = data_description.annotations_dir

    if data_description.feature_version is not None:
        # if feature_version is defined we use features
        # right now this selects all features, todo: add some argument tile_size to overwrite this

        image_feature, feature_description = db_manager.get_image_features_by_image_and_feature_version(
            image.id, data_description.feature_version
        )

        if image_feature is None:
            # Directly return the initialized dictionary with None values
            return image_info

        # Update the dictionary with the actual values
        image_info.update(get_relevant_feature_info_from_record(image_feature, data_description, feature_description))

        return image_info

    else:
        # Update the dictionary with the actual values
        image_info.update(get_relevant_image_info_from_record(image, data_description, annotations_root, stage))

        return image_info


def datasets_from_data_description(
    db_manager: DataManager,
    data_description: DataDescription,
    transform: Callable[[TileSample], RegionFromWsiDatasetSample] | None,
    stage: str,
) -> Generator[TiledWsiDataset, None, None]:
    logger.info(f"Reading manifest from {data_description.manifest_database_uri} for stage {stage}")

    assert isinstance(stage, str), "Stage should be a string."

    patients = db_manager.get_records_by_split(
        manifest_name=data_description.manifest_name,
        split_version=data_description.split_version,
        split_category=stage,
    )

    for patient_idx, patient in enumerate(patients):
        patient_labels = get_labels_from_record(patient)

        for image in patient.images:
            image_labels = get_labels_from_record(image)
            labels = None if patient_labels is image_labels is None else (patient_labels or []) + (image_labels or [])

            image_info = get_image_info(db_manager, data_description, image, stage)

            image_path = image_info["image_path"]

            if image_path is None:
                # if no feature is found...
                continue

            mpp = image_info["mpp"]
            tile_size = image_info["tile_size"]
            tile_overlap = image_info["tile_overlap"]
            backend = image_info["backend"]
            overwrite_mpp = image_info["overwrite_mpp"]
            tile_mode = image_info["tile_mode"]
            output_tile_size = image_info["output_tile_size"]
            mask = image_info["mask"]
            mask_threshold = image_info["mask_threshold"]
            rois = image_info["rois"]
            annotations = image_info["annotations"]

            assert isinstance(image_path, Path)
            assert isinstance(mpp, float)
            assert (
                isinstance(tile_size, tuple)
                and len(tile_size) == 2
                and all(isinstance(i, int) for i in tile_size)  # pylint: disable=not-an-iterable
            )
            assert (
                isinstance(tile_overlap, tuple)
                and len(tile_overlap) == 2
                and all(isinstance(i, int) for i in tile_overlap)  # pylint: disable=not-an-iterable
            )
            assert backend is not None
            assert isinstance(overwrite_mpp, float) or overwrite_mpp is None
            assert isinstance(tile_mode, TilingMode)
            assert (
                isinstance(output_tile_size, tuple)
                and len(output_tile_size) == 2
                and all(isinstance(i, int) for i in output_tile_size)  # pylint: disable=not-an-iterable
                or (output_tile_size is None)
            )
            assert isinstance(mask, WsiAnnotations) or (mask is None) or isinstance(mask, SlideImage)
            assert isinstance(mask_threshold, float) or mask_threshold is None
            assert isinstance(annotations, WsiAnnotations) or (annotations is None)

            dataset = TiledWsiDataset.from_standard_tiling(
                path=image_path,
                mpp=mpp,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                grid_order=GridOrder.C,
                tile_mode=tile_mode,
                crop=False,
                mask=mask,
                mask_threshold=mask_threshold,
                output_tile_size=output_tile_size,
                rois=rois if rois is not None else None,
                annotations=annotations if stage != "predict" else None,
                labels=labels,  # type: ignore
                transform=transform,
                backend=backend,  # type: ignore
                overwrite_mpp=(overwrite_mpp, overwrite_mpp),
                limit_bounds=True,
                apply_color_profile=data_description.apply_color_profile,
                internal_handler="vips",
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


def open_db(database_uri: str, ensure_exists: bool = True) -> Session:
    """Open a database connection.

    Parameters
    ----------
    database_uri : str
        The URI of the database.
    ensure_exists : bool, default=True
        Whether to raise an exception of the database does not exist.

    Returns
    -------
    Session
        The database session.
    """
    engine = create_engine(database_uri)

    if not ensure_exists:
        # Create tables if they don't exist
        create_tables(engine)
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
