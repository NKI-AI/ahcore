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


# TODO: Make parameterized tests and think of edge cases.
# TODO: Add a case where there are no features to write.

@pytest.mark.parametrize("num_samples, feature_size, grid_size", [
    (1, 786, (1, 1)),   # Single sample with typical feature size
    (25, 786, (5, 5)),  # Multiple samples with specific grid size
    (1, 0, (1, 1)),     # Edge case: Empty feature
])
def test_h5_tile_feature_writer(temp_h5_file: Path, num_samples: int, feature_size: int, grid_size: tuple[int, int]) -> None:
    writer = H5TileFeatureWriter(filename=temp_h5_file, size=grid_size, num_samples=num_samples)

    # Generate deterministic coordinates based on grid size
    dummy_coordinates = np.array([[x, y] for x in range(grid_size[0]) for y in range(grid_size[1])])
    if feature_size > 0:
        dummy_features = np.random.rand(num_samples, feature_size).astype(np.float32)
    else:
        # For the case with no features, create an array with the correct shape but no content
        dummy_features = np.empty((num_samples, 0), dtype=np.float32)
    tile_index = np.arange(num_samples)

    # Define a generator for the test
    def feature_generator() -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        for i in range(num_samples):
            yield dummy_coordinates[i:i + 1], dummy_features[i:i + 1] if feature_size > 0 else np.array([[]], dtype=np.float32)

    # Act: Write data to the H5 file
    writer.consume(feature_generator())

    # Assert
    with h5py.File(temp_h5_file, "r") as h5file:
        if num_samples > 0 and feature_size > 0:
            assert np.allclose(h5file["data"][:], dummy_features), "Feature data does not match expected values."
            assert np.allclose(h5file["coordinates"][:], dummy_coordinates), "Coordinates data does not match expected values."
            assert np.all(np.diff(h5file["tile_indices"][:]) >= 0), "Tile indices are not in ascending order."
        else:
            # Assert that the dataset is empty or has the expected shape with no data
            assert "data" not in h5file or h5file["data"].shape[0] == 0, "Data dataset should be empty or not exist."
            assert "coordinates" not in h5file or h5file["coordinates"].shape[0] == 0, "Coordinates dataset should be empty or not exist."
            assert "tile_indices" not in h5file or h5file["tile_indices"].shape[0] == 0, "Tile indices dataset should be empty or not exist."


# Run the tests
if __name__ == "__main__":
    pytest.main()
