import os
from typing import Any, Generator

import dotenv
import hydra
from dlup.data.dataset import TiledWsiDataset
from omegaconf import OmegaConf
from pymilvus import Collection, connections
from torch.utils.data import DataLoader

from ahcore.data.dataset import ConcatDataset
from ahcore.utils.data import DataDescription
from ahcore.utils.manifest import DataManager
from ahcore.utils.vector_database.utils import calculate_overlap, datasets_from_data_description

dotenv.load_dotenv(override=True)


def connect_to_milvus(host: str, port: int, alias: str, user: str, password: str) -> None:
    connections.connect(alias=alias, host=host, port=port, user=user, password=password)


def load_data_description(file_path: str) -> DataDescription:
    config = OmegaConf.load(file_path)
    data_description = hydra.utils.instantiate(config)
    return data_description


def create_dataset_iterator(data_description: DataDescription) -> Generator[TiledWsiDataset, None, None]:
    data_manager = DataManager(data_description.manifest_database_uri)
    datasets = datasets_from_data_description(data_description, data_manager)
    return datasets


def construct_dataset(data_iterator: Generator) -> ConcatDataset:
    datasets = []
    for idx, ds in enumerate(data_iterator):
        datasets.append(ds)
    return ConcatDataset(datasets=datasets)


def construct_dataloader(dataset: ConcatDataset, num_workers: int, batch_size: int) -> None:
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)


def query_annotated_vectors(
    data_description: DataDescription, collection: Collection, min_overlap: float = 0.25
) -> list[list[float]]:
    dataset_iterator = create_dataset_iterator(data_description=data_description)
    dataset = construct_dataset(dataset_iterator)
    dataloader = construct_dataloader(dataset, num_workers=0, batch_size=1)

    tile_sizes = data_description.inference_grid.tile_size
    tile_size = tile_sizes[0]
    vectors = []
    for i, data in enumerate(dataloader):
        # One entry at the time (we don't need to query a lot; these are just the reference vectors)
        filename, coordinate_x, coordinate_y = data["filename"][0], int(data["coordinate_x"]), int(data["coordinate_y"])
        res = query_vector(
            collection, filename, coordinate_x, coordinate_y, tile_size=tile_size, min_overlap=min_overlap
        )
        vectors += res


def create_index(collection: Collection, index_params: dict[str, Any] | None = None) -> None:
    if index_params is None:
        index_params = {
            "index_type": "FLAT",
            "metric_type": "L2",
            "params": {"nlist": 2048},
        }
    collection.create_index(field_name="embedding", index_params=index_params)


def query_vector(
    collection: Collection,
    filename: str,
    coordinate_x: int,
    coordinate_y: int,
    tile_size: int,
    min_overlap: float = 0.25,
) -> list[list[str, Any]]:
    # Define the coordinate range for the query
    min_x = coordinate_x - tile_size
    max_x = coordinate_x + tile_size
    min_y = coordinate_y - tile_size
    max_y = coordinate_y + tile_size

    # Query to find all tiles within the coordinate range for the specified filename
    expr = (
        f"filename == '{filename}' and coordinate_x >= {min_x} and coordinate_x <= {max_x} "
        f"and coordinate_y >= {min_y} and coordinate_y <= {max_y}"
    )

    # Execute the query
    results = collection.query(
        expr=expr,
        output_fields=["filename", "coordinate_x", "coordinate_y", "tile_size", "mpp", "embedding"],
        consistency_level="Strong",
    )

    # List to store embeddings with sufficient overlap
    relevant_embeddings = []

    # Check overlap for each result
    for result in results:
        overlap = calculate_overlap(
            coordinate_x, coordinate_y, tile_size, result["coordinate_x"], result["coordinate_y"], result["tile_size"]
        )

        if overlap > min_overlap:
            relevant_embeddings.append(result["embedding"])

    return relevant_embeddings


if __name__ == "__main__":
    data_description = load_data_description(os.environ.get("DATA_DESCRIPTION_PATH"))
    connect_to_milvus(
        host=os.environ.get("MILVUS_HOST"),
        user=os.environ.get("MILVUS_USER"),
        password=os.environ.get("MILVUS_PASSWORD"),
        port=os.environ.get("MILVUS_PORT"),
        alias=os.environ.get("MILVUS_ALIAS"),
    )
    collection = Collection(name="path_fomo_cls_debug_coll", using="ahcore_milvus_vector_db")
    create_index(collection)
    collection.load()
    query_annotated_vectors(data_description=data_description, collection=collection)
