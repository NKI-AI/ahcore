### THis will first be a stream of consciousness type of file which we will later clear up to their respective locations.
from typing import Any
from pymilvus import connections, Collection


ANNOTATIONS_DIR="/processing/e.marcus/susankomen_data/masks/"


def connect_to_milvus(host:str, port:int, alias: str, user: str, password: str) -> None:
    connections.connect(alias=alias, host=host, port=port, user=user, password=password)


def create_index(collection: Collection, index_params: dict[str, Any] | None = None) -> None:
    if index_params is None:
        index_params = {"index_type": "FLAT",
                        "metric_type": "L2",
                        "params": {"nlist": 2048},
                    }
    collection.create_index(field_name="embedding", index_params=index_params)


def query_vectors(collection: Collection, filenames = list[str] | str) -> list[dict[str, Any]]:
    # We need to loop through all annotations of a certain type, find the coordinates, filenames, and labels
    # We then do a vector lookup for all the filesnames and coordinates, obtaining the vector entries
    if type(filenames) == str:
        filenames = [filenames]

    # example
    res = collection.query(
    expr = f"filename in {filenames} and coordinate_x > 20000 and coordinate_x < 40000 and coordinate_y == 0",
    output_fields=["filename", "coordinate_x", "coordinate_y"], 
    consistency_level="Strong",
    limit=10,
    )
    return res


if __name__ == "__main__":
    connect_to_milvus(host="gorgophone", port=19530, alias="ahcore_milvus_vector_db", user="root", password="taart123!")
    collection = Collection(name="path_fomo_cls_debug_coll", using="ahcore_milvus_vector_db")
    collection.load()
    create_index(collection)
    result = query_vectors(collection, filenames=["images/K102490.svs", "images/K102491.svs"])
    print(result)
    print('test')
