import os
from pathlib import Path
from typing import Any

import dotenv
import matplotlib.pyplot as plt
import numpy as np
from dlup import SlideImage
from dlup.backends import ImageBackend
from dlup.data.dataset import TiledWsiDataset
from dlup.tiling import GridOrder, TilingMode
from pymilvus import Collection, connections, utility

from ahcore.transforms.pre_transforms import PreTransformTaskFactory
from ahcore.utils.data import DataDescription
from ahcore.utils.manifest import DataManager, _get_rois, parse_annotations_from_record

dotenv.load_dotenv(override=True)


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
                output["coordinates_x"] = sample["coordinates"][0]
                output["coordinates_y"] = sample["coordinates"][1]
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
