from pathlib import Path
from typing import Any, Callable, Sequence, Union, Optional, Type, overload

import numpy as np
import collections
import functools
# import pyvips
from dlup.data.dataset import (
    BaseWsiDataset,
    TileSample,
    RegionFromWsiDatasetSample,
    parse_rois,
    MaskTypes,
    _AnnotationTypes,
    _LabelTypes,
)
from dlup.tiling import Grid, GridOrder, TilingMode
from dlup.types import PathLike, ROIType

from ahcore.readers import FileImageReader, StitchingMode, ZarrFileImageReader
from ahcore.utils.types import FMEmbedType, GenericNumberArray


LRU_CACHE_SIZE = 32


@functools.lru_cache(LRU_CACHE_SIZE)
def _get_cached_file_image_reader(
    path: Path, reader: Type[FileImageReader], stitching_mode: StitchingMode = StitchingMode.NONE, **kwargs: Any
) -> "Type[FileImageReader]":
    return reader.from_file_path(filename=path, stitching_mode=stitching_mode)


class TileFeatureSample(TileSample):
    # image: pyvips.Image  # Reading image is now a lot of i/o overhead
    features: GenericNumberArray  # Embeddings features for tiles


class BaseWsiFeatureDataset(BaseWsiDataset):
    def __init__(
        self,
        path: PathLike,
        features_path: PathLike,
        regions: collections.abc.Sequence[tuple[float, float, int, int, float]],
        reader: Type[FileImageReader] = ZarrFileImageReader,
        stitching_mode: StitchingMode = StitchingMode.NONE,
        mask: MaskTypes | None = None,
        mask_threshold: float | None = 0.0,
        annotations: list[_AnnotationTypes] | _AnnotationTypes | None = None,
        labels: list[tuple[str, _LabelTypes]] | None = None,
    ):
        self._feature_path = features_path
        self._reader = reader
        self._stitching_mode = stitching_mode

        # TODO: DLUP branch feature/feature-embedding-dataset contains exclude_image_data is only on this branch to 
        # reduce i/o overhead
        # TODO: Create Backend from FileImageReader instead of this work around. 
        super().__init__(
            path=path,
            regions=regions,
            mask=mask,
            mask_threshold=mask_threshold,
            annotations=annotations,
            labels=labels,
            # exclude_image_data=True,
        )

        # For now only allow direct reading of features at scaling of 1
        file_reader = self.file_image_reader
        for region in self.regions:
            _, _, _, _, mpp = region
            _scaling = file_reader.get_scaling(mpp)
            if _scaling != 1.0:
                raise NotImplementedError("Only direct feature reading is supported as of now. Use scaling 1.0")

    @property
    def file_image_reader(self) -> "FileImageReader":
        """Mock function to FileImageReader as SlideImage.

        Returns
        -------
        FileImageReader
            _description_
        """
        return _get_cached_file_image_reader(path=self.path, reader=self._reader, stitching_mode=self._stitching_mode)

    @overload
    def __getitem__(self, index: int) -> TileFeatureSample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TileFeatureSample]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[TileFeatureSample, Sequence[TileFeatureSample]]:
        if isinstance(index, slice):
            # handle slicing
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step or 1)]

        sample = super().__getitem__(index)
        coordinates = sample["coordinates"]
        region_size = (sample["image"].width, sample["image"].height)

        # Feature embedding for a single tile will have shape similar to (196, 768), (768) or (2, 768)
        # This will correspond to the region size
        _file_image_reader = self.file_image_reader
        scaling = _file_image_reader.get_scaling(sample["mpp"])
        features = _file_image_reader.read_region(coordinates, scaling, region_size)
        sample: TileFeatureSample = {
            "features": features,
            **sample
        }
        return sample


class TiledWsiFeatureDataset(BaseWsiFeatureDataset):
    def __init__(
        self,
        path: PathLike,
        features_path: PathLike,
        grids: list[tuple[Grid, tuple[int, int], float]],
        mask: MaskTypes | None = None,
        mask_threshold: float | None = 0.0,
        annotations: list[_AnnotationTypes] | _AnnotationTypes | None = None,
        labels: list[tuple[str, _LabelTypes]] | None = None,
        transform: Callable[[TileSample], RegionFromWsiDatasetSample] | None = None,
        reader: Type[FileImageReader] = ZarrFileImageReader,
        stitching_mode: StitchingMode = StitchingMode.NONE,
    ):
        self._grids = grids
        self._transform = transform

        raise NotImplementedError
        regions = None
        super().__init__(
            path=path,
            features_path=features_path,
            regions=regions,
            mask=mask,
            mask_threshold=mask_threshold,
            annotations=annotations,
            labels=labels,
            reader=reader,
            stitching_mode=stitching_mode,
        )

    @classmethod
    def from_cached_tiling(
        cls,
        path: PathLike,
        features_path: PathLike,
        mpp: float | None,
        tile_size: tuple[int, int],
        tile_overlap: tuple[int, int],
        output_tile_size: tuple[int, int] | None = None,
        tile_mode: TilingMode = TilingMode.overflow,
        grid_order: GridOrder = GridOrder.C,
        crop: bool = False,
        mask: MaskTypes | None = None,
        mask_threshold: float | None = 0.0,
        rois: list[ROIType] | None = None,
        annotations: _AnnotationTypes | None = None,
        labels: list[tuple[str, _LabelTypes]] | None = None,
        transform: Callable[[TileFeatureSample], RegionFromWsiDatasetSample] | None = None,
        limit_bounds: bool = True,
        reader: Type[FileImageReader] = ZarrFileImageReader,
        stitching_mode: StitchingMode = StitchingMode.NONE,
    ) -> "TiledWsiFeatureDataset":
        # Assert same cached image has the same Grid that data description generates
        with reader.from_file_path(features_path, stitching_mode=stitching_mode) as file_image_reader:
            assert file_image_reader.mpp == mpp
            assert file_image_reader._metadata.get("tile_size") == tile_size
            assert file_image_reader._metadata.get("tile_overlap") == tile_overlap
            assert file_image_reader._metadata.get("tiling_mode") == tile_mode.value
            assert file_image_reader._metadata.get("grid_order") == grid_order.value

            # Get ROIs
            # scaling = original_mpp / mpp
            # if rois is not None:
            #     # Use native_size multiplied with native_mpp / mpp or something
            #     slide_level_size = file_image_reader.get_scaled_size(1.0, limit_bounds=False)

            #     # We are assuming ROIs are at original mpp still. Rescale them
            #     _rois = parse_rois(rois, slide_level_size, scaling=scaling)
            # elif limit_bounds:
            #     # This is the same as size. Limit bounds will be taken care of
            #     _rois = [file_image_reader.get_scaled_slide_bounds(scaling=1.0)]
            # else:
            #     # Use native_size again
            #     slide_level_size = file_image_reader.get_scaled_size(1.0, limit_bounds=False)
            #     _rois = [((0, 0), slide_level_size)]

        # Create grid from metadata

        # Initialize class object
        # return cls(
        #     path=path,
        #     grids=grids,
        #     mask=mask,
        #     mask_threshold=mask_threshold,
        #     annotations=annotations,
        #     labels=labels,
        #     reader=reader,
        #     stitching_mode=stitching_mode,
        # )
