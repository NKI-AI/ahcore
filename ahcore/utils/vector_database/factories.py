from enum import Enum, auto
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from pymilvus import CollectionSchema, DataType, FieldSchema

PrepareDataType: TypeAlias = Tuple[List[List[float]], List[str], List[int], List[int]]


class CollectionType(Enum):
    PATH_FOMO_CLS = auto()
    PATH_FOMO_CLS_PATCH = auto()


class MilvusCollectionFactory:
    # TODO create actual schema definitions
    def __init__(self) -> None:
        self.schema_definitions: Dict[CollectionType, Callable[[], CollectionSchema]] = {
            CollectionType.PATH_FOMO_CLS: self._path_fomo_cls,
            CollectionType.PATH_FOMO_CLS_PATCH: self._path_fomo_cls_patch,
        }

    def _path_fomo_cls(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="random", dtype=DataType.DOUBLE),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8),
        ]
        return CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")

    def _path_fomo_cls_patch(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="path_fomo_cls", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="path_fomo_embeddings", dtype=DataType.FLOAT_VECTOR, dim=8),
        ]
        return CollectionSchema(fields, "path_fomo_collection")

    def get_schema(self, collection_type: CollectionType) -> CollectionSchema:
        schema_method = self.schema_definitions.get(collection_type)
        if schema_method:
            return schema_method()
        raise ValueError("Unknown Collection Type")


class PrepareDataFactory:
    """Factory class to create data preparation methods for different collection types."""

    def __init__(self) -> None:
        self.factory_methods: Dict[CollectionType, Callable[[torch.Tensor, Dict[str, Any]], PrepareDataType]] = {
            CollectionType.PATH_FOMO_CLS: self.prepare_data_path_fomo_cls,
        }

    def get_preparer(
        self, collection_type: CollectionType
    ) -> Callable[[torch.Tensor, Dict[str, Any]], PrepareDataType]:
        if collection_type in self.factory_methods:
            return self.factory_methods[collection_type]
        else:
            raise ValueError("Unknown Collection Type for data preparation")

    def prepare_data_path_fomo_cls(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> PrepareDataType:
        """Prepare data specifically for PATH_FOMO_CLS collection."""
        embeddings = outputs.detach().cpu().numpy()
        metadata = batch["metadata1"]  # Assume 'metadata1' is where the metadata is stored
        coordinates = batch["coordinates"]  # Assume 'coordinates' holds a tuple of (x, y) for each entry

        # Convert each part of data to the appropriate list format
        embeddings_list = [embedding.tolist() for embedding in embeddings]  # List of lists of floats for embeddings
        metadata_list = list(metadata)  # List of strings for metadata
        coordinate_x_list = [coord[0] for coord in coordinates]  # List of integers for x-coordinates
        coordinate_y_list = [coord[1] for coord in coordinates]  # List of integers for y-coordinates

        return embeddings_list, metadata_list, coordinate_x_list, coordinate_y_list
