"""
This module contains writer classes. Currently implemented:

Two writer classes that can write files based on iterators, for instance, the output of a dataset class.

- `H5FileImageWriter`
- `ZarrFileImageWriter`

"""

import abc
import io
import json
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Generator, NamedTuple, Optional, Type

import h5py
import numcodecs
import numpy as np
import numpy.typing as npt
import PIL.Image
import zarr
from dlup.tiling import Grid, GridOrder, TilingMode

from ahcore.utils.io import get_logger
from ahcore.utils.types import GenericNumberArray, InferencePrecision

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


@contextmanager
def generic_file_manager(writer: Any, mode: str = "w") -> Any:
    file = writer.open_file(mode=mode)
    try:
        yield file
    finally:
        writer.close_file(file)


class WriterMetadata(NamedTuple):
    mode: str
    format: str | None
    num_channels: int
    dtype: str


class Writer(abc.ABC):
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

        self._grid_coordinates: Optional[npt.NDArray[np.int_]] = None
        self._grid_offset: npt.NDArray[np.int_] | None = None

        self._current_index: int = 0
        self._tiles_seen = 0

        self._tile_indices: Any
        self._data: Any
        self._coordinates_dataset: Any

        self._partial_suffix: str = f"{self._filename.suffix}.partial"

    @abc.abstractmethod
    def open_file(self, mode: str = "w") -> Any:
        pass

    @abc.abstractmethod
    def close_file(self, file: Any) -> Any:
        pass

    @abc.abstractmethod
    def insert_data(self, batch: GenericNumberArray) -> None:
        """Insert a batch into a dataset."""

    @abc.abstractmethod
    def create_dataset(
        self,
        file: Any,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
        compression: str,
        chunks: Optional[tuple[int, ...]] = None,
        data: Optional[GenericNumberArray] = None,
    ) -> Any:
        """Create a dataset with the given specifications."""

    @abc.abstractmethod
    def create_variable_length_dataset(
        self,
        file: Any,
        name: str,
        shape: tuple[int, ...],
        compression: str,
        chunks: Optional[tuple[int, ...] | bool] = None,
    ) -> Any:
        """Create a dataset with the given specifications."""
        pass

    @staticmethod
    def _batch_generator(
        first_coordinates_batch: tuple[GenericNumberArray, GenericNumberArray],
        batch_generator: Generator[tuple[GenericNumberArray, GenericNumberArray], None, None],
    ) -> Generator[tuple[GenericNumberArray, GenericNumberArray], None, None]:
        # We yield the first batch too so the progress bar takes the first batch also into account
        yield first_coordinates_batch
        for tile in batch_generator:
            if tile is None:
                break
            yield tile

    def get_writer_metadata(self, first_batch: GenericNumberArray) -> WriterMetadata:
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

        _dtype = str(first_batch.dtype)

        return WriterMetadata(_mode, _format, _num_channels, _dtype)

    def set_grid(self) -> None:
        # TODO: We only support a single Grid
        # TODO: Probably you can decipher the grid from the coordinates
        # TODO: This would also support multiple grids
        # TODO: One would need to collect the coordinates and based on the first and the last
        # TODO: of a single grid, one can determine the grid, and the empty indices.

        if self._grid is None:  # During validation, the grid is passed as a parameter
            grid = Grid.from_tiling(
                self._grid_offset,  # type: ignore
                size=self._size,
                tile_size=self._tile_size,
                tile_overlap=self._tile_overlap,
                mode=TilingMode.overflow,
                order=GridOrder.C,
            )
            self._grid = grid

    # TODO: Any is not the right one here, use TypedDict
    def construct_metadata(self, writer_metadata: WriterMetadata) -> dict[str, Any]:
        assert self._grid
        # This only works when the mode is 'overflow' and in 'C' order.
        metadata = {
            "mpp": self._mpp,
            "size": (int(self._size[0]), int(self._size[1])),
            "num_channels": writer_metadata.num_channels,
            "num_samples": self._num_samples,
            "tile_size": tuple(self._tile_size),
            "tile_overlap": tuple(self._tile_overlap),
            "num_tiles": len(self._grid),
            "grid_order": "C",
            "tiling_mode": "overflow",
            "mode": writer_metadata.mode,
            "format": writer_metadata.format,
            "dtype": writer_metadata.dtype,
            "is_binary": self._is_compressed_image,
            "precision": self._precision.value if self._precision else str(InferencePrecision.FP32),
            "multiplier": (
                self._precision.get_multiplier() if self._precision else InferencePrecision.FP32.get_multiplier()
            ),
            "has_color_profile": self._color_profile is not None,
        }

        if self._extra_metadata:
            metadata.update(self._extra_metadata)

        return metadata

    def adjust_batch_precision(self, batch: GenericNumberArray) -> GenericNumberArray:
        """Adjusts the batch precision based on the precision set in the writer."""
        if self._precision:
            multiplier = self._precision.get_multiplier()
            batch = batch * multiplier
            batch = batch.astype(self._precision.value)
        return batch

    @abc.abstractmethod
    def write_metadata(self, metadata: dict[str, Any], file: Any) -> None:
        """Write metadata to the file"""

    def consume(self, batch_generator: Generator[tuple[GenericNumberArray, GenericNumberArray], None, None]) -> None:
        """Consumes tiles one-by-one from a generator and writes them to the file."""
        grid_counter = 0

        try:
            with generic_file_manager(self) as file:
                first_coordinates, first_batch = next(batch_generator)
                init_writer_batch = self.adjust_batch_precision(first_batch)

                metadata = self.init_writer(first_coordinates, init_writer_batch, file)
                self.write_metadata(metadata, file)

                # Mostly for mypy
                assert self._grid, "Grid is not initialized"
                assert self._tile_indices, "Tile indices are not initialized"
                assert self._data, "Dataset is not initialized"
                assert self._coordinates_dataset, "Coordinates dataset is not initialized"

                # Need to predeclare them to make sure they are correctly written, we found this could cause issues with
                # zarr if we don't do it this way and write directly to self._tile_indices.
                tile_indices = np.full((len(self._grid),), -1, dtype=int)
                coordinates_dataset = np.full((self._num_samples, 2), -1, dtype=int)

                batch_generator = self._batch_generator((first_coordinates, first_batch), batch_generator)
                # progress bar will be used if self._progress is not None
                if self._progress:
                    batch_generator = self._progress(batch_generator, total=self._num_samples)

                for idx, (coordinates, batch) in enumerate(batch_generator):
                    self._tiles_seen += batch.shape[0]
                    batch = self.adjust_batch_precision(batch)
                    # We take a coordinate, and step through the grid until we find it.
                    # Note that this assumes that the coordinates come in C-order, so we will always hit it
                    for curr_idx, curr_coordinates in enumerate(coordinates):
                        # As long as our current coordinates are not equal to the grid coordinates, we make a step
                        while not np.all(curr_coordinates == self._grid[grid_counter]):
                            grid_counter += 1
                        # If we find it, we set it to the index, so we can find it later on
                        # This can be tested by comparing the grid evaluated at a grid index with the tile index
                        # mapped to its coordinates
                        tile_indices[grid_counter] = self._current_index + curr_idx
                        grid_counter += 1

                    batch_size = batch.shape[0]
                    coordinates_dataset[self._current_index : self._current_index + batch_size] = coordinates

                    if self._is_compressed_image:
                        # When the batch has variable lengths, we need to insert each sample separately
                        for sample in batch:
                            self.insert_data(sample[np.newaxis, ...])
                            self._current_index += 1
                    else:
                        self.insert_data(batch)
                        self._current_index += batch_size

                self._tile_indices[:] = tile_indices
                self._coordinates_dataset[:] = coordinates_dataset

        except Exception as e:
            logger.error("Error in consumer thread for %s: %s", self._filename, e, exc_info=e)

        else:
            # When done writing rename the file.
            self._filename.with_suffix(self._partial_suffix).rename(self._filename)

    def init_writer(self, first_coordinates: GenericNumberArray, first_batch: GenericNumberArray, file: Any) -> Any:
        """Initializes the image_dataset based on the first tile."""

        writer_metadata = self.get_writer_metadata(first_batch)

        self._current_index = 0
        # The grid can be smaller than the actual image when slide bounds are given.
        # As the grid should cover the image, the offset is given by the first tile.
        self._grid_offset = np.array(first_coordinates[0])

        self._coordinates_dataset = self.create_dataset(
            file, name="coordinates", shape=(self._num_samples, 2), dtype=np.int_, compression="gzip"
        )

        self.set_grid()
        assert self._grid
        num_tiles = len(self._grid)

        self._tile_indices = self.create_dataset(
            file,
            name="tile_indices",
            shape=(num_tiles,),
            dtype=np.int_,
            compression="gzip",
        )

        if not self._is_compressed_image:
            shape = first_batch.shape[1:]
            self._data = self.create_dataset(
                file,
                "data",
                shape=(self._num_samples,) + shape,
                dtype=first_batch.dtype,
                compression="gzip",
                chunks=(1,) + shape,
            )
        else:
            self._data = self.create_variable_length_dataset(
                file,
                name="data",
                shape=(self._num_samples,),
                chunks=(1,),
                compression="gzip",
            )

        if self._color_profile:
            data = (np.frombuffer(self._color_profile, dtype=np.uint8),)
            self.create_dataset(
                file,
                name="color_profile",
                shape=(data[0].size,),
                compression="gzip",
                dtype="uint8",
            )
            file["color_profile"][:] = data

        metadata = self.construct_metadata(writer_metadata)

        return metadata


class ZarrFileImageWriter(Writer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._store = zarr.ZipStore(str(self._filename.with_suffix(self._partial_suffix)), mode="w")

    def open_file(self, mode: str = "w") -> zarr.Group:
        if mode == "w":
            zarr_group = zarr.group(store=self._store, overwrite=True)
        else:
            zarr_group = zarr.open_group(store=self._store, mode=mode)
        return zarr_group

    def close_file(self, file: Any) -> None:
        # file.close()
        self._store.close()

    def create_dataset(
        self,
        file: Any,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
        compression: str,
        chunks: Optional[tuple[int, ...]] = None,
        data: Optional[GenericNumberArray] = None,
    ) -> Any:
        """Create a Zarr dataset.

        Note: Do not use the `data` parameter when you're going to overwrite it later on, that will give warnings that
        files are overwritten in the zip.
        """
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2) if compression == "gzip" else None
        dataset = file.create_dataset(name, data=data, shape=shape, dtype=dtype, chunks=chunks, compressor=compressor)
        return dataset

    def create_variable_length_dataset(
        self,
        file: Any,
        name: str,
        shape: tuple[int, ...],
        compression: str,
        chunks: Optional[tuple[int, ...] | bool] = None,
    ) -> Any:
        """Create a dataset with the given specifications."""

        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2) if compression == "gzip" else None

        assert shape
        dataset = file.create_dataset(
            name,
            shape=shape[0],
            dtype=object,
            object_codec=numcodecs.vlen.VLenArray(dtype="uint8"),
            compressor=compressor,
            chunks=chunks,
        )
        return dataset

    def insert_data(self, batch: GenericNumberArray) -> None:
        """Insert a batch into a Zarr dataset."""
        if not batch.shape[0] == 1 and self._is_compressed_image:
            raise ValueError(f"Batch should have a single element when writing zarr. Got batch shape {batch.shape}.")

        if self._is_compressed_image:
            self._data[self._current_index] = batch.reshape(-1)
        else:
            self._data[self._current_index : self._current_index + batch.shape[0]] = (
                batch.flatten() if self._is_compressed_image else batch
            )

    def write_metadata(self, metadata: dict[str, Any], file: Any) -> None:
        """Write metadata to Zarr group attributes."""
        file.attrs.update(metadata)

    def add_associated_images(
        self,
        images: tuple[tuple[str, npt.NDArray[np.uint8]], ...],
        description: Optional[str] = None,
    ) -> None:
        raise NotImplementedError("Associated images are not yet supported for Zarr files.")


class H5FileImageWriter(Writer):
    """Image writer that writes tile-by-tile to h5."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._data: h5py.Dataset
        self._coordinates_dataset: Optional[h5py.Dataset] = None
        self._tile_indices: Optional[h5py.Dataset] = None

    def open_file(self, mode: str = "w") -> Any:
        return h5py.File(self._filename.with_suffix(self._partial_suffix), mode)

    def close_file(self, file: Any) -> None:
        file.close()

    def write_metadata(self, metadata: Any, file: Any) -> None:
        metadata_json = json.dumps(metadata)
        file.attrs["metadata"] = metadata_json

    def insert_data(self, batch: GenericNumberArray) -> None:
        """Insert a batch into a H5 dataset."""
        if not batch.shape[0] == 1 and self._is_compressed_image:
            raise ValueError(f"Batch should have a single element when writing h5. Got batch shape {batch.shape}.")
        batch_size = batch.shape[0]
        self._data[self._current_index : self._current_index + batch_size] = (
            batch.flatten() if self._is_compressed_image else batch
        )

    def create_dataset(
        self,
        h5file: h5py.File,
        name: str,
        shape: tuple[int, ...] | None,
        dtype: Any,
        compression: str | None,
        chunks: Optional[Optional[tuple[int, ...]] | bool] = None,
        data: Optional[GenericNumberArray] = None,
    ) -> Any:
        if chunks is None:
            chunks = True  # Use HDF5's auto-chunking

        return h5file.create_dataset(name, data=data, shape=shape, dtype=dtype, compression=compression, chunks=chunks)

    def create_variable_length_dataset(
        self,
        h5file: h5py.File,
        name: str,
        shape: tuple[int, ...],
        compression: str | None,
        chunks: Optional[Optional[tuple[int, ...]] | bool] = None,
        data: Optional[GenericNumberArray] = None,
    ) -> Any:
        dt = h5py.vlen_dtype(np.dtype("uint8"))  # Variable-length uint8 data type
        return self.create_dataset(h5file, name, shape=shape, dtype=dt, compression=compression, chunks=chunks)

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
