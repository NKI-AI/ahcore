import pytest
import h5py
import numpy as np
from ahcore.writers import H5FileImageWriter


@pytest.fixture
def temp_h5_file(tmp_path):
    h5_file_path = tmp_path / "test_data.h5"
    yield h5_file_path


def test_h5_file_image_writer(temp_h5_file):
    # Test parameters
    size = (1000, 800)
    mpp = 0.5
    tile_size = (200, 200)
    tile_overlap = (50, 50)
    num_samples = 10

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
    dummy_coordinates = np.random.randint(0, 255, (10, 2))
    dummy_batch = np.random.rand(10, 3, 200, 200)

    # Initialize the writer
    with h5py.File(temp_h5_file, "w") as h5file:
        writer.init_writer(dummy_coordinates, dummy_batch, h5file)

    # Use a generator to yield dummy batches
    def batch_generator():
        for i in range(num_samples):
            yield (dummy_coordinates, dummy_batch)

    # Write data to the H5 file
    writer.consume(batch_generator())

    # Perform assertions
    with h5py.File(temp_h5_file, "r") as h5file:
        assert "data" in h5file
        assert "coordinates" in h5file
        assert "tile_indices" in h5file
        assert "metadata" in h5file.attrs
        # Optionally add more specific assertions here


# Run the tests
if __name__ == "__main__":
    pytest.main()
