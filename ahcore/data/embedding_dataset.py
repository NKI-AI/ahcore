from ahcore.readers import ZarrFileImageReader, FileImageReader, StitchingMode
from ahcore.utils.types import FMEmbedType
from pathlib import Path
from typing import Dict, Any, Type, Optional
from torch.utils.data import Dataset
from dlup.tiling import Grid, TilingMode, GridOrder


class BaseEmbeddingDataset(Dataset):
    def __init__(self, filename: Path, classification_label: list, reader_class:  Type[FileImageReader]) -> None:
        """
        Base class for datasets of feature embeddings.

        Parameters
        ----------
        filename: Path
            The path to the cache file containing the feature embeddings. This could be an H5 file or a Zarr file.

        classification_label: list
            Slide level labels for the embedding dataset. Each embedding will be associated with a single label.

        reader_class: Type[FileImageReader]
            The reader class to use to read the feature embeddings, e.g., H5FileImageReader or ZarrFileImageReader.
        """
        super().__init__()
        self._filename = filename
        self._wsi_classification_label = classification_label
        self._reader_class = reader_class
        self._embedding_type: FMEmbedType = FMEmbedType.CLS_TOKEN

        self._file: Optional[Any] = None
        self._dataset_metadata = None
        self._mpp = None
        self._num_embeddings = None
        self._feature_length = None
        self._dtype = None

    @property
    def wsi_labels(self) -> list:
        return self._wsi_classification_label

    def init_dataset(self):
        with self._reader_class(self._filename, stitching_mode=StitchingMode.CROP) as cache_reader:
            self._mpp = cache_reader.mpp
            self._num_embeddings = cache_reader._metadata["num_samples"]



    @property
    def embedding_type(self) -> FMEmbedType:
        return self._embedding_type

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class WSIEmbeddingDataset(BaseEmbeddingDataset):
    def __init__(self, filename: Path, classification_label: list, reader_class: Type[FileImageReader]) -> None:
        """
        Dataset class for feature
        """
        super().__init__(filename, classification_label, reader_class)
        self.init_dataset()

    def __getitem__(self, index):
        pass

    def __len__(self):
        raise NotImplementedError




if __name__ == '__main__':
    dataset = WSIEmbeddingDataset(Path('/processing/a.karkala/feature_extraction/2024-05-21_22-21-53/outputs/AhcoreJitModel/0_0/0ec1eb99ad42782386bb8e24d0604775be2389e3deb9916ef0d010a4ea7c1d06.cache')
                                  , classification_label = [0], reader_class=ZarrFileImageReader)
    print(dataset[0])