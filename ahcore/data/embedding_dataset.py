from pathlib import Path
from typing import Any, Dict, Optional, Type, TypedDict

import numpy as np
from dlup.tiling import Grid, GridOrder, TilingMode
from torch.utils.data import Dataset

from ahcore.readers import FileImageReader, StitchingMode, ZarrFileImageReader
from ahcore.utils.types import FMEmbedType, GenericNumberArray


class WSIEmbeddingSample(TypedDict):
    wsi_embeddings: GenericNumberArray  # placeholder


class BaseEmbeddingDataset(Dataset):
    def __init__(self, filename: Path, classification_label: list, reader_class: Type[FileImageReader]) -> None:
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

        self.__empty_feature: GenericNumberArray | None = None

        self._embedding_type = None
        self._embed_mode = None
        self._wsi_metadata = None
        self._num_embeddings = None
        self._embedding_length = None
        self._grid = None

        self._init_dataset()
        self._regions = self._generate_regions()

    @property
    def wsi_labels(self) -> list:
        return self._wsi_classification_label

    @property
    def feature_length(self) -> int:
        return self._embedding_length

    @property
    def embedding_type(self) -> FMEmbedType:
        return self._embedding_type

    def _empty_feature(self) -> GenericNumberArray:
        """
        This method should return an empty feature of the same dtype as the embeddings in the dataset.

        Returns
        -------
        GenericNumberArray
            An empty feature of the same dtype as the embeddings in the dataset.
        """
        if self.__empty_feature is not None:
            return self.__empty_feature

        # When this happens we would already be in the read_region, and self._num_channels would be populated.
        assert self._embedding_length

        self.__empty_feature = np.zeros(self._embedding_length, dtype=self._wsi_metadata["dtype"])
        return self.__empty_feature

    def _init_dataset(self) -> None:
        """
        Initialize the feature embedding dataset.
        -------
        None
        """
        # This method should set the following attributes:

        # self._embedding_type
        # self._wsi_metadata
        # self._num_embeddings
        # self._embedding_length
        # self._grid

        # With a context manager, we read the cache file to get the metadata of the WSI.
        # We then set the self._wsi_metadata attribute to the metadata of the WSI.
        # We then set the self._num_embeddings attribute to the number of embeddings in the dataset.
        # We then set the self._embedding_length attribute to the length of the embeddings.
        # The self._embedding_type attribute should be set based on the shape of the embeddings.
        # We should initialize the self._grid attribute here using the size parameter found in the metadata.
        pass

    def _get_feature_embedding(self, location: tuple[int, int]) -> Any:
        raise NotImplementedError

    def _generate_regions(self) -> list[tuple[int, int]]:
        """
        This function should generate the coordinates of the regions for which the embeddings are available.
        A careful consideration should be given to coordinates which maybe filled with "empty_tile".
        """
        # We loop the coordinates of the grid and check if the tile is empty. If it is not empty, we add the
        # coordinates to the regions list.
        pass

    def __len__(self):
        return self._num_embeddings

    def __getitem__(self, idx):
        raise NotImplementedError


class WSIEmbeddingDataset(BaseEmbeddingDataset):
    def __init__(self, filename: Path, classification_label: list, reader_class: Type[FileImageReader]) -> None:
        """
        Dataset class for feature
        """
        super().__init__(filename, classification_label, reader_class)
        """
        The following attributes are initialized in the constructor:
        - self._filename: The path to the cache file containing the feature embeddings.
        - self._wsi_classification_label: Slide level labels for the embedding dataset. This could be in the manifest we
         generate.
        - self._reader_class: The reader used to read the feature embeddings. This could be in the manifest we generate.
        - self._embedding_type: The type of embedding to return. This is inferred based on the shape of features.
        - self._wsi_metadata: The metadata of the WSI.
        - self._num_embeddings: The number of embeddings in the dataset.
        - self._embedding_length: The length of the embeddings.
        """
        pass

    def _get_feature_embedding(self, location: tuple[int, int]) -> Any:
        """
        This method should return the feature embeddings for a given region (size) starting from the defined location
        in the WSI. The embed_type should be used to determine the type of embedding to return. For example, if the
        embed_type is CLS_TOKEN, the method should return the CLS tokens for the given region.
        Check ahcore.utils.types to understand more about FMEmbedTypes.
        One flag we could add is the mode in which the embeddings are returned. For example, they could be 2D or 3D.
        """
        # First, there is a call made to reader class method via context managers to read the embeddings.
        # This can be achieved using read_region_raw() method of the reader class. Note that the size will be (1,1).
        # If the self._embedding_type is FMEmbedType.CLS_TOKEN, the method should return the CLS tokens for the given
        # region. If the self._embedding_type is FMEmbedType.PATCH_TOKEN, the method should return the PATCH tokens in
        # the right mode specified by self._embed_mode. This is either a 2D or 3D array.
        pass

    def __getitem__(self, index) -> WSIEmbeddingSample:
        coordinates = self._regions[index]
        sample: WSIEmbeddingSample = {"wsi_embeddings": self._get_feature_embedding(location=coordinates)}
        return sample
