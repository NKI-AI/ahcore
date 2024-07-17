import os
from typing import Any

import dotenv
import numpy as np
import numpy.typing as npt
import shapely
from dlup import SlideImage
from dlup.annotations import AnnotationClass
from dlup.annotations import Polygon as DlupPolygon
from dlup.annotations import WsiAnnotations
from pymilvus import Collection, connections
from pymilvus.client.abstract import Hit, SearchResult
from shapely import box

from ahcore.utils.data import DataDescription
from ahcore.utils.vector_database.plot_utils import plot_wsi_and_annotation_overlay
from ahcore.utils.vector_database.utils import (
    calculate_overlap,
    construct_dataloader,
    construct_dataset,
    create_dataset_iterator,
    generate_filenames,
    load_data_description,
)

dotenv.load_dotenv(override=True)


def connect_to_milvus(host: str, port: int, alias: str, user: str, password: str) -> None:
    connections.connect(alias=alias, host=host, port=port, user=user, password=password)


def query_annotated_vectors(
    data_description: DataDescription, collection: Collection, min_overlap: float = 0.25, force_recompute: bool = False
) -> list[float]:
    cache_location = "/processing/e.marcus/average_annotated_vector.npy"
    if not force_recompute and os.path.exists(cache_location):
        return list(np.load(cache_location))

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
        if i % 100 == 0:
            print(f"Processed {i} entries")

    average_vector = [sum(x) / len(x) for x in zip(*vectors)]
    np.save(cache_location, average_vector)

    return average_vector


def create_index(collection: Collection, index_params: dict[str, Any] | None = None) -> None:
    if index_params is None:
        # index_params = {
        #     "index_type": "FLAT",
        #     "metric_type": "L2",
        #     "params": {"nlist": 2048},
        # }
        index_params = {
            "index_type": "FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 8192},
        }
    collection.create_index(field_name="embedding", index_params=index_params)


def query_vector(
    collection: Collection,
    filename: str,
    coordinate_x: int,
    coordinate_y: int,
    tile_size: int,
    min_overlap: float = 0.25,
) -> list[list[float]]:
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
        consistency_level="Eventually",
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


def search_vectors_in_radius(
    collection: Collection,
    reference_vector: list[float],
    limit_results: int = 10,
    search_radius: float = 1.0,
    range_filter: float = 0.0,
) -> list[list[str, Any]]:
    # Define the search parameters
    # param = {
    #     "metric_type": "L2",
    #     "params": {"radius": search_radius, "range_filter": range_filter},  # Below this, the results are not returned
    # }
    param = {
        "metric_type": "COSINE",
        "params": {
            "radius": search_radius,  # Radius of the search circle
            "range_filter": range_filter,  # Range filter to filter out vectors that are not within the search circle
        },
    }

    # Execute the query
    results = collection.search(
        data=[reference_vector],
        anns_field="embedding",
        param=param,
        limit=limit_results,
        output_fields=["filename", "coordinate_x", "coordinate_y", "tile_size", "mpp", "embedding"],
    )

    return results


def extract_results_per_wsi(search_results: SearchResult) -> dict[str, list[Hit]]:
    """Extracts search results per WSI. Note this is only used for testing purposes on a small dataset."""
    hits = search_results[0]
    hits_per_wsi = {}
    for hit in hits:
        filename = hit.filename
        if filename not in hits_per_wsi:
            hits_per_wsi[filename] = []
        hits_per_wsi[filename].append(hit)
    return hits_per_wsi


def compute_iou_from_hits(image: SlideImage, annotation: WsiAnnotations, hits: list[Hit]) -> float:
    """Computes IoU between a hit and the annotations"""
    total_union_area = 0.0
    total_intersection_area = 0.0
    scaling = image.get_scaling(0.5)  # We inference at 0.5 mpp

    for hit in hits:
        x, y, width, height = hit.coordinate_x, hit.coordinate_y, hit.tile_size, hit.tile_size
        location = (x, y)

        def _affine_coords(coords: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
            return coords * scaling - np.asarray(location, dtype=np.float_)

        rect = box(x, y, x + width, y + height)
        rect = shapely.transform(rect, _affine_coords)  # Transform the coordinates
        annotations = annotation.read_region((x, y), scaling, (width, height))
        # if not annotations:
        #     print("test")

        local_intersection_area = 0
        for local_annotation in annotations:
            local_intersection = local_annotation.intersection(rect)
            local_intersection_area += local_intersection.area
            if local_intersection.area == 0:
                print("test")

        if local_intersection_area == 0:  # No intersect between box and annotations
            total_union_area += 2 * rect.area
        else:
            total_intersection_area += local_intersection_area
            total_union_area += 2 * rect.area - local_intersection_area

    total_iou = total_intersection_area / total_union_area if total_union_area != 0 else 0
    return total_iou


def create_wsi_annotation_from_hits(hits: list[Hit]) -> WsiAnnotations:
    """Creates a WsiAnnotations object from the hits"""
    polygon_list = []

    for hit in hits:
        x, y, width, height = hit.coordinate_x, hit.coordinate_y, hit.tile_size, hit.tile_size
        rect = box(x, y, x + width, y + height)
        polygon = DlupPolygon(rect, a_cls=AnnotationClass(label="hits", annotation_type="POLYGON"))
        polygon_list.append(polygon)

    annotations = WsiAnnotations(layers=polygon_list)

    return annotations


def compute_iou_per_wsi(search_results: SearchResult, data_dir: str, annotations_dir: str) -> dict[str, float]:
    """Computes total IoU for each separate WSI found in the search results"""
    hits_per_wsi: dict[str, list[Hit]] = extract_results_per_wsi(search_results)
    iou_per_wsi = {}

    for filename, hits in hits_per_wsi.items():
        image_filename, annotations_filename = generate_filenames(filename, data_dir, annotations_dir)
        hit_annotations = create_wsi_annotation_from_hits(hits)
        image = SlideImage.from_file_path(image_filename)
        annotation = WsiAnnotations.from_geojson(annotations_filename)
        plot_wsi_and_annotation_overlay(
            image, annotation, hit_annotations, mpp=20, tile_size=(20, 20), filename_appendage=filename
        )
        # iou = compute_iou_from_hits(image, annotation, hits)
        # iou_per_wsi[filename] = iou
    return iou_per_wsi


if __name__ == "__main__":
    data_description = load_data_description(os.environ.get("DATA_DESCRIPTION_PATH"))
    connect_to_milvus(
        host=os.environ.get("MILVUS_HOST"),
        user=os.environ.get("MILVUS_USER"),
        password=os.environ.get("MILVUS_PASSWORD"),
        port=os.environ.get("MILVUS_PORT"),
        alias=os.environ.get("MILVUS_ALIAS"),
    )
    # collection_name = "path_fomo_cls_debug_coll"
    collection_name = "path_fomo_5wsi_annotated_test"
    collection = Collection(name=collection_name, using="ahcore_milvus_vector_db")
    create_index(collection)
    collection.load()
    average_annotated_vector = query_annotated_vectors(
        data_description=data_description, collection=collection, min_overlap=0.5
    )
    search_results = search_vectors_in_radius(
        collection,
        reference_vector=average_annotated_vector,
        limit_results=10000,
        search_radius=0.65,
        range_filter=1.0,
    )
    iou_per_wsi = compute_iou_per_wsi(
        search_results, data_dir=data_description.data_dir, annotations_dir=data_description.annotations_dir
    )
    print(iou_per_wsi)
