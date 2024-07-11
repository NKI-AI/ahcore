import random
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

from pymilvus import CollectionSchema, DataType, FieldSchema

PrepareDataTypePathCLS: TypeAlias = Tuple[List[str], List[int], List[int], List[int], List[float], List[List[float]]]
PrepareDataTypeDebug: TypeAlias = Tuple[List[float], List[List[float]]]


class CollectionType(Enum):
    PATH_FOMO_CLS = auto()
    PATH_FOMO_CLS_PATCH = auto()
    DEBUG = auto()


class MilvusCollectionFactory:
    def __init__(self, embedding_dim: int) -> None:
        self.schema_definitions: Dict[CollectionType, Callable[[], CollectionSchema]] = {
            CollectionType.PATH_FOMO_CLS: self._path_fomo_cls,
            CollectionType.PATH_FOMO_CLS_PATCH: self._path_fomo_cls_patch,
            CollectionType.DEBUG: self._debug,
        }
        self._embedding_dim = embedding_dim
        self._max_string_length = 250  # mandatory to put a limit

    def _path_fomo_cls(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=self._max_string_length),
            FieldSchema(name="coordinate_x", dtype=DataType.INT64),
            FieldSchema(name="coordinate_y", dtype=DataType.INT64),
            FieldSchema(name="tile_size", dtype=DataType.INT64),
            FieldSchema(name="mpp", dtype=DataType.FLOAT),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._embedding_dim),
        ]
        return CollectionSchema(fields, "pathology fomo CLS embedding collection")

    def _path_fomo_cls_patch(self) -> CollectionSchema:
        raise NotImplementedError

    def _debug(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="random", dtype=DataType.DOUBLE),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8),
        ]
        return CollectionSchema(fields, "a debug collection")

    def get_schema(self, collection_type: CollectionType) -> CollectionSchema:
        schema_method = self.schema_definitions.get(collection_type)
        if schema_method:
            return schema_method()
        raise ValueError(f"Unknown Collection Type {collection_type}")


class PrepareDataFactory:
    """Factory class to create data preparation methods for different collection types."""

    def __init__(self) -> None:
        self.factory_methods: Dict[
            CollectionType, Callable[[Dict[str, Any], Dict[str, Any]], PrepareDataTypePathCLS | PrepareDataTypeDebug]
        ] = {
            CollectionType.PATH_FOMO_CLS: self._prepare_data_path_fomo_cls,
            CollectionType.DEBUG: self._prepare_data_debug,
        }

    def get_preparer(
        self, collection_type: CollectionType
    ) -> Callable[[Dict[str, Any], Dict[str, Any]], PrepareDataTypePathCLS | PrepareDataTypeDebug]:
        if collection_type in self.factory_methods:
            return self.factory_methods[collection_type]
        else:
            raise ValueError("Unknown Collection Type for data preparation")

    def _prepare_data_path_fomo_cls(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> PrepareDataTypePathCLS:
        """Prepare data specifically for PATH_FOMO_CLS collection."""
        image = batch["image"]
        predictions = outputs["prediction"]
        unprocessed_fnames = [Path(fname) for fname in outputs["path"]]
        prepared_fnames = [f"{local_fname.name}" for local_fname in unprocessed_fnames]
        prepared_coordinates_x = outputs["coordinates"][0].cpu().numpy().tolist()
        prepared_coordinates_y = outputs["coordinates"][1].cpu().numpy().tolist()
        prepared_tile_sizes = [image.shape[-1] for i in range(image.shape[0])]  # assumes square tiles
        prepared_mpps = outputs["mpp"].cpu().numpy().tolist()
        prepared_embeddings = predictions.cpu().numpy().tolist()

        return (
            prepared_fnames,
            prepared_coordinates_x,
            prepared_coordinates_y,
            prepared_tile_sizes,
            prepared_mpps,
            prepared_embeddings,
        )

    def _prepare_data_debug(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> PrepareDataTypeDebug:
        """Prepare data specifically for DEBUG collection."""
        entities = (
            [float(random.randrange(-20, -10)) for _ in range(3000)],
            [[random.random() for _ in range(8)] for _ in range(3000)],
        )
        return entities
