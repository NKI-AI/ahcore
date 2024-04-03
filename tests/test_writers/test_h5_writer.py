from pathlib import Path
from typing import Generator

import h5py
import numpy as np
import pytest

from ahcore.utils.types import GenericNumberArray
from ahcore.writers import H5FileImageWriter


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
    def batch_generator() -> Generator[tuple[GenericNumberArray, GenericNumberArray], None, None]:
        for i in range(num_samples):
            yield dummy_coordinates, dummy_batch

    # Write data to the H5 file
    writer.consume(batch_generator())

    # Perform assertions
    with h5py.File(temp_h5_file, "r") as h5file:
        assert "data" in h5file
        assert "coordinates" in h5file
        assert np.array(h5file["data"]).shape == (num_samples, 3, 200, 200)
        assert np.array(h5file["coordinates"]).shape == (num_samples, 2)
        assert np.allclose(h5file["data"], dummy_batch)
        assert np.allclose(h5file["coordinates"], dummy_coordinates)


# Run the tests
if __name__ == "__main__":
    pytest.main()
