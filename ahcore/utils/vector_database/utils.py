import os
from pathlib import Path
from typing import Any, Generator

import dotenv
import hydra
import matplotlib.pyplot as plt
import numpy as np
from dlup import SlideImage
from dlup.backends import ImageBackend
from dlup.data.dataset import TiledWsiDataset
from dlup.tiling import GridOrder, TilingMode
from omegaconf import OmegaConf
from pymilvus import Collection, connections, utility
from pymilvus.client.abstract import Hit
from torch.utils.data import DataLoader

from ahcore.data.dataset import ConcatDataset
from ahcore.transforms.pre_transforms import PreTransformTaskFactory
from ahcore.utils.data import DataDescription
from ahcore.utils.manifest import DataManager, _get_rois, parse_annotations_from_record

dotenv.load_dotenv(override=True)


def calculate_overlap(x1: int, y1: int, size1: int, x2: int, y2: int, size2: int) -> float:
    # Calculate the intersection coordinates
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + size1, x2 + size2)
    iy2 = min(y1 + size1, y2 + size2)

    # Calculate intersection area
    intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    # Calculate union area
    union_area = 2 * size1 * size1 - intersection_area

    # Return the overlap ratio
    return intersection_area / union_area if union_area != 0 else 0


def plot_tile(hit: Hit) -> None:
    """Plots a tile from a SlideImage."""
    filename, x, y, width, height, mpp = (
        hit.filename,
        hit.coordinate_x,
        hit.coordinate_y,
        hit.tile_size,
        hit.tile_size,
        hit.mpp,
    )
    data_dir = os.environ.get("DATA_DIR")
    filename = Path(data_dir) / filename

    image = SlideImage.from_file_path(filename)
    scaling = image.get_scaling(mpp)
    tile = image.read_region((x, y), scaling, (width, height))
    tile = tile.numpy().astype(np.uint8)
    plt.imshow(tile)
    plt.savefig("/home/e.marcus/projects/ahcore/debug_tile_plots/tile.png")


def delete_collection(collection_name: str) -> None:
    """Deletes a collection by name."""
    connections.connect(
        host=os.environ.get("MILVUS_HOST"),
        user=os.environ.get("MILVUS_USER"),
        password=os.environ.get("MILVUS_PASSWORD"),
        port=os.environ.get("MILVUS_PORT"),
        alias=os.environ.get("MILVUS_ALIAS"),
    )
    if utility.has_collection(collection_name, using=os.environ.get("MILVUS_ALIAS")):
        collection = Collection(name=collection_name, using=os.environ.get("MILVUS_ALIAS"))
        collection.drop()
        print(f"Collection {collection_name} deleted.")
    else:
        print(f"Collection {collection_name} does not exist.")


class CoordinateCollate:
    def __call__(self, sample) -> dict[str, Any]:
        output = {}
        for key in sample:
            if key == "path":
                output["filename"] = str(Path(sample["path"]).name)
            if key == "coordinates":
                output["coordinate_x"] = sample["coordinates"][0]
                output["coordinate_y"] = sample["coordinates"][1]
            if key == "image":
                output["image"] = sample["image"]
            if key == "mpp":
                output["mpp"] = sample["mpp"]

        return output


def datasets_from_data_description(data_description: DataDescription, db_manager: DataManager):
    stage: str = "predict"  # Always true for this pipeline

    image_root = data_description.data_dir
    annotations_dir = data_description.annotations_dir
    grid_description = data_description.inference_grid

    transforms = PreTransformTaskFactory([CoordinateCollate()])  # Includes default pre transforms

    patients = db_manager.get_records_by_split(
        manifest_name=data_description.manifest_name,
        split_version=data_description.split_version,
        split_category=stage,
    )
    for patient in patients:
        for image in patient.images:
            annotations = parse_annotations_from_record(annotations_root=annotations_dir, record=image.annotations)
            annotations.filter(data_description.roi_name)
            rois = _get_rois(annotations, data_description, stage)
            mask_threshold = data_description.mask_threshold if data_description.mask_threshold is not None else 0.0

            dataset = TiledWsiDataset.from_standard_tiling(
                path=image_root / image.filename,
                mpp=grid_description.mpp,
                tile_size=grid_description.tile_size,
                tile_overlap=grid_description.tile_overlap,
                tile_mode=TilingMode.skip,
                grid_order=GridOrder.C,
                crop=False,
                mask=None,
                mask_threshold=mask_threshold,
                output_tile_size=getattr(grid_description, "output_tile_size", None),
                rois=rois,
                annotations=None,
                labels=None,
                transform=transforms,
                backend=ImageBackend[str(image.reader)],
                overwrite_mpp=(image.mpp, image.mpp),
                limit_bounds=True,
                apply_color_profile=data_description.apply_color_profile,
                internal_handler="vips",
            )

            yield dataset


def construct_dataloader(dataset: ConcatDataset, num_workers: int, batch_size: int) -> None:
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)


def construct_dataset(data_iterator: Generator) -> ConcatDataset:
    datasets = []
    for idx, ds in enumerate(data_iterator):
        datasets.append(ds)
    return ConcatDataset(datasets=datasets)


def create_dataset_iterator(data_description: DataDescription) -> Generator[TiledWsiDataset, None, None]:
    data_manager = DataManager(data_description.manifest_database_uri)
    datasets = datasets_from_data_description(data_description, data_manager)
    return datasets


def load_data_description(file_path: str) -> DataDescription:
    config = OmegaConf.load(file_path)
    data_description = hydra.utils.instantiate(config)
    return data_description


def generate_filenames(filename: str, data_dir: str, annotations_dir: str) -> tuple[str, str]:
    """Generates the image and annotations filenames from the WSI filename"""
    image_filename = Path(data_dir) / filename
    annotations_filename = Path(annotations_dir) / filename.replace(".svs", ".svs.geojson")
    return image_filename, annotations_filename
