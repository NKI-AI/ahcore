from pathlib import Path
from typing import Generator

import h5py
import numpy as np
import pytest

from ahcore.utils.types import GenericArray
from ahcore.writers import H5FileImageWriter, H5TileFeatureWriter


@pytest.fixture
def temp_h5_file(tmp_path: Path) -> Generator[Path, None, None]:
    h5_file_path = tmp_path / "test_data.h5"
    yield h5_file_path


def test_h5_file_image_writer(temp_h5_file: Path) -> None:
    """
    This test the H5FileImageWriter class for the following case:

    Assuming that we have a tile of size (200, 200) and we want to write it to an H5 file.
    This test writes the tile to the H5 file and then reads it back to perform assertions.

    Parameters
    ----------
    temp_h5_file

    Returns
    -------
    None
    """
    size = (200, 200)
    mpp = 0.5
    tile_size = (200, 200)
    tile_overlap = (0, 0)
    num_samples = 1

    # Create an instance of H5FileImageWriter
    writer = H5FileImageWriter(
        filename=temp_h5_file,
        size=size,
        mpp=mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        num_samples=num_samples,
    )

    # Dummy batch data for testing
    dummy_coordinates = np.random.randint(0, 255, (1, 2))
    dummy_batch = np.random.rand(1, 3, 200, 200)

    # Use a generator to yield dummy batches
    def batch_generator() -> Generator[tuple[GenericArray, GenericArray], None, None]:
        for i in range(num_samples):
            yield dummy_coordinates, dummy_batch

    # Write data to the H5 file
    writer.consume(batch_generator())

    # Perform assertions
    with h5py.File(temp_h5_file, "r") as h5file:
        assert "data" in h5file
        assert "coordinates" in h5file
        assert h5file["data"].shape == (num_samples, 3, 200, 200)
        assert h5file["coordinates"].shape == (num_samples, 2)
        assert np.allclose(h5file["data"], dummy_batch)
        assert np.allclose(h5file["coordinates"], dummy_coordinates)


def test_h5_tile_feature_writer(temp_h5_file: Path) -> None:
    """
    This test the H5TileFeatureWriter class for the following case.

    Assume that we have an image of size (1024, 1024) tiled up in a batch of 4 tiles of size (512, 512).
    The feature dimension is 1024. So, we have 4 feature vectors of size 1024 which need to be written to the H5 file.
    We want to do this while retaining their spatial orientation in correspondence with the spatial orientation of the tiles.

    The features should be written in the following order:
    1. Top left tile   -> Top left feature
    2. Top right tile  -> Top right feature
    3. Bottom left tile -> Bottom left feature
    4. Bottom right tile -> Bottom right feature

    Parameters
    ----------
    temp_h5_file : Path

    Returns
    -------
    None
    """
    size = (2, 2)
    feature_dimension = 1024
    num_features = 4

    writer = H5TileFeatureWriter(filename=temp_h5_file, size=size)

    coords = np.stack([[0, 0], [0, 1], [1, 0], [1, 1]], 0)
    features = np.random.rand(num_features, feature_dimension)

    def feature_generator(
        coords: GenericArray, features: GenericArray
    ) -> Generator[tuple[GenericArray, GenericArray], None, None]:
        for coord, feature in zip(coords, features):
            yield coord, feature

    # Write data to the H5 file
    writer.consume_features(feature_generator(coords, features))

    # Perform assertions
    with h5py.File(temp_h5_file, "r") as h5file:
        assert "tile_feature_vectors" in h5file
        assert h5file["tile_feature_vectors"].shape == (size[0], size[1], feature_dimension)
        assert np.allclose(h5file["tile_feature_vectors"][0, 0, :], features[0])
        assert np.allclose(h5file["tile_feature_vectors"][0, 1, :], features[1])
        assert np.allclose(h5file["tile_feature_vectors"][1, 0, :], features[2])
        assert np.allclose(h5file["tile_feature_vectors"][1, 1, :], features[3])


# Run the tests
if __name__ == "__main__":
    pytest.main()
