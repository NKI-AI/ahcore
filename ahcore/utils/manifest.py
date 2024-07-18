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
from dlup.backends import ImageBackend
from dlup.data.dataset import RegionFromWsiDatasetSample, TiledWsiDataset, TileSample
from dlup.tiling import GridOrder, TilingMode
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import exists

from ahcore.backends import ImageCacheBackend
from ahcore.exceptions import RecordNotFoundError
from ahcore.utils.data import DataDescription
from ahcore.utils.database_models import (
    Base,
    CacheDescription,
    CategoryEnum,
    Image,
    ImageAnnotations,
    ImageCache,
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

    def get_split_definition(
        self,
        manifest_name: str,
        split_version: str,
    ) -> SplitDefinitions:
        """Fetch the split definition based on manifest name and split version.

        Parameters
        ----------
        manifest_name : str
            The name of the manifest.
        split_version : str
            Fetch the metadata for an image based on its filename.

        Returns
        -------
        SplitDefinitions
            The associated SplitDefinitions of the manifest and split version.

        Raises
        ------
        RecordNotFoundError
            Error gets raised when manifest or split version does not exist.
        """
        manifest = self._session.query(Manifest).filter_by(name=manifest_name).first()
        try:
            self._ensure_record(manifest, f"Manifest with name {manifest_name}")
        except RecordNotFoundError as e:
            raise RecordNotFoundError(
                f"Manifest with name {manifest_name} not found. "
                f"Available manifest names: {', '.join([str(m.name) for m in self._session.query(Manifest).all()])}"
            ) from e

        assert manifest is not None
        split_definition = (
            self._session.query(SplitDefinitions).filter_by(manifest_id=manifest.id, version=split_version).first()
        )
        self._ensure_record(split_definition, f"Split definition with version {split_version}")

        # This is because mypy is complaining otherwise,
        # but _ensure_record effectively ensures that the record is not None
        assert split_definition is not None
        return split_definition

    def get_patients_by_split(
        self,
        split_definition: SplitDefinitions,
        split_category: Optional[str] = None,
    ) -> Generator[Patient, None, None]:
        """Yields the patients for a given split definition, and optional split category.

        Parameters
        ----------
        split_definition : SplitDefinitions
            The definition of the split.
        split_category : Optional[str], optional
            The category of the split. Must be a string compatible with `CategoryEnum`. If `split_category=None`, all
            patients in the split definition will be returned, by default None

        Yields
        ------
        Generator[Patient, None, None]
            The patients in the given split defintion (and split category)
        """
        query = (
            self._session.query(Patient)
            .join(Split)
            .filter(
                Patient.manifest_id == split_definition.manifest_id,
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
        split_definition = self.get_split_definition(manifest_name=manifest_name, split_version=split_version)
        for patient in self.get_patients_by_split(split_definition, split_category):
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

    def get_cache_description(
        self,
        split_definition: SplitDefinitions,
    ) -> Optional[CacheDescription]:
        """Fetch the cache description from a split definition if it exists. If no cache description is associated with
        the split definition, None is returned.

        Parameters
        ----------
        split_definition : SplitDefinitions
            The definition of the split.

        Returns
        -------
        Optional[CacheDescription]
            Associated CacheDescription from SplitDefinitions object, if it exists.
        """
        return (
            self._session.query(CacheDescription)
            .filter(CacheDescription.split_definition_id == split_definition.id)
            .first()
        )

    def get_image_cache(self, image: Image, cache_description: CacheDescription) -> Optional[ImageCache]:
        """Fetch image cache from image and cache description.

        Parameters
        ----------
        image : Image
            Image to fetch cache from.
        cache_description : CacheDescription
            CacheDescription for ImageCache

        Returns
        -------
        Optional[ImageCache]
            Optionally, returns ImageCache of the Image if it exists.
        """
        return (
            self._session.query(ImageCache).filter(
                ImageCache.image_id == image.id, ImageCache.cache_description_id == cache_description.id
            )
        ).first()

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

    split_definition = db_manager.get_split_definition(
        data_description.manifest_name,
        split_version=data_description.split_version,
    )
    patients = db_manager.get_patients_by_split(
        split_definition=split_definition,
        split_category=stage,
    )

    crop = False
    cache_description = db_manager.get_cache_description(split_definition=split_definition)
    if cache_description is not None:
        logger.info(
            f"CacheDescription found for SplitDefinition {split_definition.version}. "
            "Checking if grid descriptions and data descriptions are compatible."
        )

        # TODO: Should not assert when we want to read all features at the same time
        assert grid_description.mpp == cache_description.mpp
        cache_tile_size = tuple(cache_description.tile_size_width, cache_description.tile_size_height)
        assert tuple(grid_description.tile_size) == cache_tile_size
        cache_tile_overlap = tuple(cache_description.tile_overlap_width, cache_description.tile_overlap_height)
        assert tuple(grid_description.tile_overlap) == cache_tile_overlap
        assert getattr(grid_description, "output_tile_size", None) is None

        # We set apply_color_profile to False because
        data_description.apply_color_profile = False
        # Crop must be true so BoundaryMode.CROP is used instead of BoundaryMode.zeros for features
        crop = True

    for patient in patients:
        patient_labels = get_labels_from_record(patient)

        for image in patient.images:
            mask, annotations = get_mask_and_annotations_from_record(annotations_root, image)
            assert isinstance(mask, WsiAnnotations) or (mask is None)
            image_labels = get_labels_from_record(image)
            labels = None if patient_labels is image_labels is None else (patient_labels or []) + (image_labels or [])
            rois = _get_rois(mask, data_description, stage)
            mask_threshold = 0.0 if stage != "fit" else data_description.mask_threshold

            original_image_mpp = image.mpp
            # Overwrite image with ImageCache if CacheDescription in SplitDefinitions.
            if cache_description is not None:
                image_cache = db_manager.get_image_cache(image=image, cache_description=cache_description)
                assert image_cache is not None

                # Partially initialize ImageCacheBackend because SlideImage does not take stitching_mode as kwarg
                backend = functools.partial(
                    ImageCacheBackend[str(image_cache.reader)].value, stitching_mode=image_cache.stitching_mode
                )
                # _extra_pixel_required only working for internal_handler "none" now.
                internal_handler = "none"

                # We have to scale the annotations to level zero of the ImageCache.
                if annotations is not None:
                    scaling = original_image_mpp / cache_description.mpp
                    annotations.apply_affine_transform_to_annotations(scaling=scaling, offset=[0, 0])

                # TODO: This takes too long to initialize from masks
                # Overwrite mask and mask_threshold to only sample tiles from annotated areas
                # mask = annotations
                # mask_threshold = 0.01

                # Overwrite image and original_image_mpp with image_cache to be used in dataset
                image = image_cache  # type: ignore
                original_image_mpp = cache_description.mpp
            else:
                backend = ImageBackend[str(image.reader)]
                internal_handler = "vips"

            dataset = TiledWsiDataset.from_standard_tiling(
                path=image_root / image.filename,
                mpp=grid_description.mpp,
                tile_size=grid_description.tile_size,
                tile_overlap=grid_description.tile_overlap,
                tile_mode=TilingMode.overflow,
                grid_order=GridOrder.C,
                crop=crop,
                mask=mask,
                mask_threshold=mask_threshold,
                output_tile_size=getattr(grid_description, "output_tile_size", None),
                rois=rois if rois is not None else None,
                annotations=annotations if stage != "predict" else None,
                labels=labels,  # type: ignore
                transform=transform,
                backend=backend,
                overwrite_mpp=(original_image_mpp, original_image_mpp),
                limit_bounds=True,
                apply_color_profile=data_description.apply_color_profile,
                internal_handler=internal_handler,
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
