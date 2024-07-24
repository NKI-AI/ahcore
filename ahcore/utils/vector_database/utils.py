import hashlib
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Generator

import dotenv
import hydra
from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.backends import ImageBackend
from dlup.data.dataset import TiledWsiDataset
from dlup.tiling import GridOrder, TilingMode
from omegaconf import OmegaConf
from pymilvus import Collection, connections, utility
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


def calculate_distance_to_annotation(annotation: WsiAnnotations, x: int, y: int, scaling: float = 1.0) -> float:
    bounding_box_xy, bounding_box_wh = annotation.bounding_box
    if scaling != 1.0:
        bounding_box_xy = (int(bounding_box_xy[0] * scaling), int(bounding_box_xy[1] * scaling))
        bounding_box_wh = (int(bounding_box_wh[0] * scaling), int(bounding_box_wh[1] * scaling))
    center_x = bounding_box_xy[0] + bounding_box_wh[0] / 2
    center_y = bounding_box_xy[1] + bounding_box_wh[1] / 2
    return ((center_x - x) ** 2 + (center_y - y) ** 2) ** 0.5


def calculate_total_annotation_area(annotation: WsiAnnotations, scaling: float) -> int:
    bounding_box_xy, bounding_box_wh = annotation.bounding_box
    poly_list = annotation.read_region(bounding_box_xy, scaling=scaling, size=bounding_box_wh)
    area = sum([poly.area for poly in poly_list])
    return area


def compute_precision_recall(
    image: SlideImage,
    annotation: WsiAnnotations,
    model_output: WsiAnnotations,
    mpp: float,
    tile_size: tuple[int, int],
    distance_cutoff: float,
) -> tuple[float, float]:
    scaling = image.get_scaling(mpp)
    distance_cutoff = distance_cutoff * scaling
    dataset_annotation = TiledWsiDataset.from_standard_tiling(
        image.identifier, mpp=mpp, tile_size=tile_size, tile_overlap=(0, 0), mask=annotation
    )

    dataset_model_output = TiledWsiDataset.from_standard_tiling(
        image.identifier, mpp=mpp, tile_size=tile_size, tile_overlap=(0, 0), mask=model_output
    )

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Convert annotations and model outputs to a set of coordinates for easier comparison
    annotation_tiles = set((d["coordinates"][0], d["coordinates"][1]) for d in dataset_annotation)
    model_output_tiles = set((d["coordinates"][0], d["coordinates"][1]) for d in dataset_model_output)

    for coords in model_output_tiles:
        if (
            calculate_distance_to_annotation(annotation, coords[0], coords[1], scaling=scaling) <= distance_cutoff
        ):  # make sure we stay within annotated region
            if coords in annotation_tiles:
                true_positives += 1
            else:
                false_positives += 1
    # Evaluate false negatives separately
    for coords in annotation_tiles:
        if coords not in model_output_tiles:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


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


def dict_to_uuid(input_dict: dict) -> uuid.UUID:
    """Create a unique identifier for a pydantic model.

    This is done by pickling the object, and computing the sha256 hash of the pickled object and converting this to
    an UUID. The UUID is generated using the sha256 hash as a namespace, ensuring similar lengths. The chance of
    a collision is astronomically small.

    Arguments
    ---------
    base_model: BaseModel
        The BaseModel to create a unique identifier for.

    Returns
    -------
    uuid.UUID
        A unique identifier for the BaseModel.
    """
    # Serialize the object
    serialized_data = pickle.dumps(input_dict)

    # Generate a sha256 hash of the serialized data
    obj_hash = hashlib.sha256(serialized_data).digest()

    # Use the hash as a namespace to generate a UUID
    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, obj_hash.hex())

    return unique_id


# if __name__ == "__main__":
#     coll_name = "debug_collection_concat"
#     delete_collection(coll_name)
