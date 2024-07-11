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
from ahcore.utils.vector_database.utils import datasets_from_data_description 

dotenv.load_dotenv(override=True)


def connect_to_milvus(host: str, port: int, alias: str, user: str, password: str) -> None:
    connections.connect(alias=alias, host=host, port=port, user=user, password=password)


def load_data_description(file_path: str) -> DataDescription:
    config = OmegaConf.load(file_path)
    data_description = hydra.utils.instantiate(config)
    return data_description


def create_dataset_iterator() -> Generator[TiledWsiDataset, None, None]:
    data_description = load_data_description(os.environ.get("DATA_DESCRIPTION_PATH"))
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


def query_annotated_vectors(collection: Collection) -> list[list[float]]:
    # We need to get the annotation, and the roi name, and convert the annotation to a mask, and then call the datasets from data description
    dataset_iterator = create_dataset_iterator()
    dataset = construct_dataset(dataset_iterator)
    dataloader = construct_dataloader(dataset, num_workers=0, batch_size=1)

    vectors = []
    for i, data in enumerate(dataloader):
        if i > 5:
            break
        print(data)
        filename, coordinate_x, coordinate_y = data["filename"], data["coordinate_x"], data["coordinate_y"]
        res = query_vector(collection, filename, coordinate_x, coordinate_y)
        vectors.append(res)


def create_index(collection: Collection, index_params: dict[str, Any] | None = None) -> None:
    if index_params is None:
        index_params = {
            "index_type": "FLAT",
            "metric_type": "L2",
            "params": {"nlist": 2048},
        }
    collection.create_index(field_name="embedding", index_params=index_params)


def query_vector(
    collection: Collection, filename: str, coordinate_x: int, coordinate_y: int, tile_size: int = 224
) -> list[dict[str, Any]]:
    pass


if __name__ == "__main__":
    query_annotated_vectors()
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
