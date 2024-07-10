import matplotlib.pyplot as plt
import numpy as np
from dlup import SlideImage
from pymilvus import Collection, connections, utility

ALIAS = "ahcore_milvus_vector_db"
MILVUS_HOST = "gorgophone"
MILVUS_PORT = 19530
MILVUS_USER = "root"
MILVUS_PASSWORD = "taart123!"


def plot_tile(filename: str, mpp: float, x: int, y: int, width: int, height: int) -> None:
    """Plots a tile from a SlideImage."""
    image = SlideImage.from_file_path(filename)
    scaling = image.get_scaling(mpp)
    tile = image.read_region((x, y), scaling, (width, height))
    tile = tile.numpy().astype(np.uint8)
    plt.imshow(tile)
    plt.savefig("/home/e.marcus/projects/ahcore/debug_tile_plots/tile.png")


def delete_collection(collection_name: str) -> None:
    """Deletes a collection by name."""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, user=MILVUS_USER, password=MILVUS_PASSWORD, alias=ALIAS)
    if utility.has_collection(collection_name, using=ALIAS):
        collection = Collection(name=collection_name, using=ALIAS)
        collection.drop()
        print(f"Collection {collection_name} deleted.")
    else:
        print(f"Collection {collection_name} does not exist.")


if __name__ == "__main__":
    delete_collection("path_fomo_cls_debug_coll")
