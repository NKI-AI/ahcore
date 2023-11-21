"""
This module contains writer classes. Currently implemented:

- `H5FileImageWriter`: class to write H5 files based on iterators, for instance, the output of a dataset
  class. Can for instance be used to store outputs of models. The `readers` contain separate modules to read these
  h5 files.

"""
import json
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Generator, Optional

import h5py
import numpy as np
import numpy.typing as npt
from dlup.tiling import Grid, GridOrder, TilingMode

from ahcore.utils.io import get_logger
from ahcore.utils.types import GenericArray

logger = get_logger(__name__)


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
            is_binary: bool = False,
            progress: Optional[Any] = None,
    ) -> None:
        self._grid_offset = None
        self._grid: Optional[Grid] = None
        self._grid_coordinates: Optional[npt.NDArray[np.int_]] = None
        self._filename: Path = filename
        self._size: tuple[int, int] = size
        self._mpp: float = mpp
        self._tile_size: tuple[int, int] = tile_size
        self._tile_overlap: tuple[int, int] = tile_overlap
        self._num_samples: int = num_samples
        self._is_binary: bool = is_binary
        self._progress = progress
        self._data: Optional[h5py.Dataset] = None
        self._coordinates_dataset: Optional[h5py.Dataset] = None
        self._tile_indices: Optional[h5py.Dataset] = None
        self._current_index: int = 0

        self._logger = logger  # maybe not the best way, think about it
        self._logger.debug("Writing h5 to %s", self._filename)

    def init_writer(self, first_coordinates: GenericArray, first_batch: GenericArray, h5file: h5py.File) -> None:
        """Initializes the image_dataset based on the first tile."""
        batch_shape = np.asarray(first_batch).shape
        batch_dtype = np.asarray(first_batch).dtype

        self._grid_offset = list(first_coordinates[0])
        self._current_index = 0

        self._coordinates_dataset = h5file.create_dataset(
            "coordinates",
            shape=(self._num_samples, 2),
            dtype=int,
            compression="gzip",
        )

        # TODO: We only support a single Grid
        grid = Grid.from_tiling(
            self._grid_offset,
            size=self._size,
            tile_size=self._tile_size,
            tile_overlap=self._tile_overlap,
            mode=TilingMode.overflow,
            order=GridOrder.C,
        )
        num_tiles = len(grid)
        self._grid = grid
        self._tile_indices = h5file.create_dataset(
            "tile_indices",
            shape=(num_tiles,),
            dtype=int,
            compression="gzip",
        )
        # Initialize to -1, which is the default value
        self._tile_indices[:] = -1

        if not self._is_binary:
            self._data = h5file.create_dataset(
                "data",
                shape=(self._num_samples,) + batch_shape[1:],
                dtype=batch_dtype,
                compression="gzip",
                chunks=(1,) + batch_shape[1:],
            )
        else:
            dt = h5py.vlen_dtype(np.dtype("uint8"))  # Variable-length uint8 data type
            self._data = h5file.create_dataset(
                "data",
                shape=(self._num_samples,),
                dtype=dt,
                chunks=(1,),
            )

        # This only works when the mode is 'overflow' and in 'C' order.
        metadata = {
            "mpp": self._mpp,
            "dtype": str(batch_dtype),
            "shape": tuple(batch_shape[1:]),
            "size": (int(self._size[0]), int(self._size[1])),
            "num_channels": batch_shape[1],
            "num_samples": self._num_samples,
            "tile_size": tuple(self._tile_size),
            "tile_overlap": tuple(self._tile_overlap),
            "num_tiles": num_tiles,
            "grid_order": "C",
            "mode": "overflow",
            "is_binary": self._is_binary,
        }
        metadata_json = json.dumps(metadata)
        h5file.attrs["metadata"] = metadata_json

    def add_associated_images(
            self,
            images: tuple[tuple[str, npt.NDArray[np.uint8]], ...],
            description: Optional[str] = None,
    ) -> None:
        """Adds associated images to the h5 file."""

        # Create a compound dataset "associated_images"
        with h5py.File(self._filename, "r+") as h5file:
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
        try:
            with h5py.File(self._filename.with_suffix(".h5.partial"), "w") as h5file:
                first_coordinates, first_batch = next(batch_generator)
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
                    # Vectorized indexing to find matching grid indices
                    grid_indices = np.where(np.all(coordinates[:, np.newaxis, :] == self._grid, axis=-1))

                    if grid_indices[0].size > 0:
                        # Update tile indices using vectorized operations
                        self._tile_indices[grid_indices[0]] = self._current_index + np.arange(len(grid_indices[0]))
                        self._tile_indices.flush()

                    batch_size = len(coordinates)
                    # Update data and coordinates datasets in batches
                    self._data[self._current_index: self._current_index + batch_size] = batch
                    self._coordinates_dataset[self._current_index: self._current_index + batch_size] = coordinates
                    self._current_index += batch_size

                    self._data.flush()
                    self._coordinates_dataset.flush()

            del self._grid
            del self._tile_indices
            del self._data
            del self._coordinates_dataset

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
