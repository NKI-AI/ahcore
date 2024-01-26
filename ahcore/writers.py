"""
This module contains writer classes. Currently implemented:

- `H5FileImageWriter`: class to write H5 files based on iterators, for instance, the output of a dataset
  class. Can for instance be used to store outputs of models. The `readers` contain separate modules to read these
  h5 files.

"""
import io
import json
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Generator, Optional

import h5py
import numpy as np
import numpy.typing as npt
import PIL.Image
from dlup.tiling import Grid, GridOrder, TilingMode

from ahcore.utils.io import get_logger
from ahcore.utils.types import GenericArray, InferencePrecision

logger = get_logger(__name__)


def decode_array_to_pil(array: npt.NDArray[np.uint8]) -> PIL.Image.Image:
    """Convert encoded array to PIL image

    Parameters
    ----------
    array : npt.NDArray[npt.uint8]
        The encoded array

    Returns
    -------
    PIL.Image.Image
        The decoded image

    """
    with io.BytesIO(array.tobytes()) as f:
        image = PIL.Image.open(f)
        # If you don't this, the image will be a reference and will be closed when exiting the context manager.
        # Any explicit copy copies the image into memory as a standard PIL.Image.Image, losing the format information.
        image.load()
    return image


class H5TileFeatureWriter:
    """Feature writer that writes tile-by-tile feature representation to h5."""

    def __init__(self, filename: Path, size: tuple[int, int]) -> None:
        self._filename = filename
        # The size corresponds to number of tiles along each coordinate axis.
        self._size = size
        self._tile_feature_dataset: Optional[h5py.Dataset] = None
        self._feature_dim: Optional[int] = None
        self._logger = logger

    def init_writer(self, first_features: GenericArray, h5file: h5py.File) -> None:
        self._feature_dim = first_features.shape[0]
        self._tile_feature_dataset = h5file.create_dataset(
            "tile_feature_vectors",
            shape=(self._size[0], self._size[1], self._feature_dim),
            dtype=first_features.dtype,
            compression="gzip",
            chunks=(1, 1, self._feature_dim),  # Chunking for each tile feature vector
        )

    def consume_features(
        self,
        feature_generator: Generator[tuple[GenericArray, GenericArray], None, None],
        connection_to_parent: Optional[Connection] = None,
    ) -> None:
        """Consumes tiles one-by-one from a generator and writes them to the h5 file."""
        try:
            with h5py.File(self._filename.with_suffix(".h5.partial"), "w") as h5file:
                first_coordinates, first_features = next(feature_generator)
                self.init_writer(first_features, h5file)

                assert self._tile_feature_dataset, "Tile feature dataset is not initialized"

                _feature_generator = self._feature_generator(first_coordinates, first_features, feature_generator)

                for tile_coordinates, feature in _feature_generator:
                    # The spatial organisation of feature vectors corresponds to the spatial organisation of the tiles.
                    self._tile_feature_dataset[tile_coordinates[0], tile_coordinates[1], :] = feature

        except Exception as e:
            self._logger.error("Error in consumer thread for %s: %s", self._filename, e, exc_info=e)
            if connection_to_parent:
                connection_to_parent.send((False, self._filename, e))
        else:
            self._filename.with_suffix(".h5.partial").rename(self._filename)

        finally:
            if connection_to_parent:
                connection_to_parent.send((True, None, None))
                connection_to_parent.close()

    @staticmethod
    def _feature_generator(
        first_feature_coordinates: GenericArray, first_features: GenericArray, feature_generator: Generator[Any, None, None]
    ) -> Generator[Any, None, None]:
        # We plug the first coordinates and the first features back into the generator
        # That way, we can yield them while h5 writing.
        yield first_feature_coordinates, first_features
        for coordinates, feature in feature_generator:
            if feature is None:
                break
            yield coordinates, feature


class H5FileImageWriter:
    """Image writer that writes tile-by-tile to h5."""

    def __init__(
        self,
        filename: Path,
        size: tuple[int, int],
        mpp: float,
        tile_size: tuple[int, int],
        tile_overlap: tuple[int, int],
        num_samples: int,
        is_compressed_image: bool = False,
        color_profile: bytes | None = None,
        progress: Optional[Any] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        precision: InferencePrecision | None = None,
        grid: Grid | None = None,
    ) -> None:
        self._grid = grid
        self._grid_coordinates: Optional[npt.NDArray[np.int_]] = None
        self._grid_offset: npt.NDArray[np.int_] | None = None
        self._filename: Path = filename
        self._size: tuple[int, int] = size
        self._mpp: float = mpp
        self._tile_size: tuple[int, int] = tile_size
        self._tile_overlap: tuple[int, int] = tile_overlap
        self._num_samples: int = num_samples
        self._is_compressed_image: bool = is_compressed_image
        self._color_profile: bytes | None = color_profile
        self._extra_metadata = extra_metadata
        self._precision = precision
        self._progress = progress
        self._data: Optional[h5py.Dataset] = None
        self._coordinates_dataset: Optional[h5py.Dataset] = None
        self._tile_indices: Optional[h5py.Dataset] = None
        self._current_index: int = 0

        self._logger = logger  # maybe not the best way, think about it
        self._logger.debug("Writing h5 to %s", self._filename)

    def init_writer(self, first_coordinates: GenericArray, first_batch: GenericArray, h5file: h5py.File) -> None:
        """Initializes the image_dataset based on the first tile."""
        if self._is_compressed_image:
            if self._precision is not None:
                raise ValueError("Precision cannot be set when writing compressed images.")
            # We need to read the first batch as it is a compressed PIL image
            _first_pil = decode_array_to_pil(first_batch[0])
            _mode = _first_pil.mode
            _format = _first_pil.format
            _num_channels = len(_first_pil.getbands())
        else:
            _mode = "ARRAY"
            _format = "RAW"
            _num_channels = first_batch.shape[1]

        self._current_index = 0
        # The grid can be smaller than the actual image when slide bounds are given.
        # As the grid should cover the image, the offset is given by the first tile.
        self._grid_offset = np.array(first_coordinates[0])

        self._coordinates_dataset = h5file.create_dataset(
            "coordinates",
            shape=(self._num_samples, 2),
            dtype=int,
            compression="gzip",
        )

        # TODO: We only support a single Grid
        # TODO: Probably you can decipher the grid from the coordinates
        # TODO: This would also support multiple grids
        # TODO: One would need to collect the coordinates and based on the first and the last
        # TODO: of a single grid, one can determine the grid, and the empty indices.
        if self._grid is None:  # During validation, the grid is passed as a parameter
            self._grid = Grid.from_tiling(
                self._grid_offset,
                size=self._size,
                tile_size=self._tile_size,
                tile_overlap=self._tile_overlap,
                mode=TilingMode.overflow,
                order=GridOrder.C,
            )

        num_tiles = len(self._grid)
        self._tile_indices = h5file.create_dataset(
            "tile_indices",
            shape=(num_tiles,),
            dtype=int,
            compression="gzip",
        )
        # Initialize to -1, which is the default value
        self._tile_indices[:] = -1

        if not self._is_compressed_image:
            shape = first_batch.shape[1:]
            self._data = h5file.create_dataset(
                "data",
                shape=(self._num_samples,) + shape,
                dtype=first_batch.dtype,
                compression="gzip",
                chunks=(1,) + shape,
            )
        else:
            dt = h5py.vlen_dtype(np.dtype("uint8"))  # Variable-length uint8 data type
            self._data = h5file.create_dataset(
                "data",
                shape=(self._num_samples,),
                dtype=dt,
                chunks=(1,),
            )

        if self._color_profile:
            h5file.create_dataset(
                "color_profile", data=np.frombuffer(self._color_profile, dtype=np.uint8), dtype="uint8"
            )

        # This only works when the mode is 'overflow' and in 'C' order.
        metadata = {
            "mpp": self._mpp,
            "size": (int(self._size[0]), int(self._size[1])),
            "num_channels": _num_channels,
            "num_samples": self._num_samples,
            "tile_size": tuple(self._tile_size),
            "tile_overlap": tuple(self._tile_overlap),
            "num_tiles": num_tiles,
            "grid_order": "C",
            "tiling_mode": "overflow",
            "mode": _mode,
            "format": _format,
            "dtype": str(first_batch.dtype),
            "is_binary": self._is_compressed_image,
            "precision": self._precision.value if self._precision else str(InferencePrecision.FP32),
            "multiplier": self._precision.get_multiplier()
            if self._precision
            else InferencePrecision.FP32.get_multiplier(),
            "has_color_profile": self._color_profile is not None,
        }

        if self._extra_metadata:
            metadata.update(self._extra_metadata)
        metadata_json = json.dumps(metadata)
        h5file.attrs["metadata"] = metadata_json

    def adjust_batch_precision(self, batch: GenericArray) -> GenericArray:
        """Adjusts the batch precision based on the precision set in the writer."""
        if self._precision:
            multiplier = self._precision.get_multiplier()
            batch = batch * multiplier
            batch = batch.astype(self._precision.value)
        return batch

    def add_associated_images(
        self,
        images: tuple[tuple[str, npt.NDArray[np.uint8]], ...],
        description: Optional[str] = None,
    ) -> None:
        """Adds associated images to the h5 file."""

        # Create a compound dataset "associated_images"
        with h5py.File(self._filename, "a") as h5file:
            associated_images = h5file.create_group("associated_images")
            for name, image in images:
                associated_images.create_dataset(name, data=image)

            if description:
                associated_images.attrs["description"] = description

    def consume(
        self,
        batch_generator: Generator[tuple[GenericArray, GenericArray], None, None],
        connection_to_parent: Optional[Connection] = None,
    ) -> None:
        """Consumes tiles one-by-one from a generator and writes them to the h5 file."""
        grid_counter = 0

        try:
            with h5py.File(self._filename.with_suffix(".h5.partial"), "w") as h5file:
                first_coordinates, first_batch = next(batch_generator)
                first_batch = self.adjust_batch_precision(first_batch)

                self.init_writer(first_coordinates, first_batch, h5file)

                # Mostly for mypy
                assert self._grid, "Grid is not initialized"
                assert self._tile_indices, "Tile indices are not initialized"
                assert self._data, "Dataset is not initialized"
                assert self._coordinates_dataset, "Coordinates dataset is not initialized"

                batch_generator = self._batch_generator((first_coordinates, first_batch), batch_generator)
                # progress bar will be used if self._progress is not None
                if self._progress:
                    batch_generator = self._progress(batch_generator, total=self._num_samples)

                for coordinates, batch in batch_generator:
                    batch = self.adjust_batch_precision(batch)
                    # We take a coordinate, and step through the grid until we find it.
                    # Note that this assumes that the coordinates come in C-order, so we will always hit it
                    for idx, curr_coordinates in enumerate(coordinates):
                        # As long as our current coordinates are not equal to the grid coordinates, we make a step
                        while not np.all(curr_coordinates == self._grid[grid_counter]):
                            grid_counter += 1
                        # If we find it, we set it to the index, so we can find it later on
                        # This can be tested by comparing the grid evaluated at a grid index with the tile index
                        # mapped to its coordinates
                        self._tile_indices[grid_counter] = self._current_index + idx
                        grid_counter += 1

                    batch_size = batch.shape[0]
                    self._data[self._current_index : self._current_index + batch_size] = batch
                    self._coordinates_dataset[self._current_index : self._current_index + batch_size] = coordinates
                    self._current_index += batch_size

        except Exception as e:
            self._logger.error("Error in consumer thread for %s: %s", self._filename, e, exc_info=e)
            if connection_to_parent:
                connection_to_parent.send((False, self._filename, e))  # Send a message to the parent
        else:
            # When done writing rename the file.
            self._filename.with_suffix(".h5.partial").rename(self._filename)
        finally:
            if connection_to_parent:
                connection_to_parent.send((True, None, None))
                connection_to_parent.close()

    @staticmethod
    def _batch_generator(
        first_coordinates_batch: Any, batch_generator: Generator[Any, None, None]
    ) -> Generator[Any, None, None]:
        # We yield the first batch too so the progress bar takes the first batch also into account
        yield first_coordinates_batch
        for tile in batch_generator:
            if tile is None:
                break
            yield tile
