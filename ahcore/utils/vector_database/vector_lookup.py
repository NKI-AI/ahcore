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
from pymilvus import Collection
from pymilvus.client.abstract import Hit, SearchResult
from shapely import box

from ahcore.utils.data import DataDescription
from ahcore.utils.vector_database.annotated_vector_querier import AnnotatedVectorQuerier
from ahcore.utils.vector_database.plot_utils import plot_wsi_and_annotation_overlay
from ahcore.utils.vector_database.utils import (
    calculate_distance_to_annotation,
    calculate_total_annotation_area,
    compute_precision_recall,
    connect_to_milvus,
    generate_filenames,
    load_data_description,
)

dotenv.load_dotenv(override=True)


def create_index(collection: Collection, index_params: dict[str, Any] | None = None) -> None:
    if index_params is None:
        index_params = {
            "index_type": "FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 8192},
        }
    collection.create_index(field_name="embedding", index_params=index_params)


def search_vectors_in_radius(
    collection: Collection,
    reference_vector: list[float],
    limit_results: int = 10,
    search_radius: float = 1.0,
    range_filter: float = 0.0,
) -> list[list[str, Any]]:
    # Define the search parameters
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
    # total_union_area = 0.0
    total_intersection_area = 0.0
    scaling = image.get_scaling(0.5)  # We inference at 0.5 mpp
    total_annotation_area = calculate_total_annotation_area(annotation, scaling)

    for hit in hits:
        x, y, width, height = hit.coordinate_x, hit.coordinate_y, hit.tile_size, hit.tile_size
        location = (x, y)

        if (
            calculate_distance_to_annotation(annotation, x, y) > 15000
        ):  # We are on the second 'slice' of the WSI where pathologist did not annotate
            continue

        def _affine_coords(coords: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
            return coords * scaling - np.asarray(location, dtype=np.float_)

        rect = box(x, y, x + width, y + height)
        rect = shapely.transform(rect, _affine_coords)  # Transform the coordinates
        annotations = annotation.read_region((x, y), scaling, (width, height))

        local_intersection_area = 0
        for local_annotation in annotations:
            local_intersection = local_annotation.intersection(rect)
            # local_intersection_area += local_intersection.area
            if local_intersection.area != 0:  # the box intersects with annotation
                local_intersection_area += local_intersection.area

        if local_intersection_area != 0:
            total_intersection_area += local_intersection_area

    total_iou = total_intersection_area / total_annotation_area if total_annotation_area != 0 else 0
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


def compute_metrics_per_wsi(
    search_results: SearchResult, data_description: DataDescription, plot_results: bool = False
) -> dict[str, float]:
    """Computes total IoU for each separate WSI found in the search results"""
    data_dir, annotations_dir = data_description.data_dir, data_description.annotations_dir
    hits_per_wsi: dict[str, list[Hit]] = extract_results_per_wsi(search_results)
    # iou_per_wsi = {}
    precision_recall_per_wsi = {}

    for filename, hits in hits_per_wsi.items():
        image_filename, annotations_filename = generate_filenames(filename, data_dir, annotations_dir)
        hit_annotations = create_wsi_annotation_from_hits(hits)
        image = SlideImage.from_file_path(image_filename, internal_handler="vips")
        annotation = WsiAnnotations.from_geojson(annotations_filename)
        annotation.filter(data_description.roi_name)
        precision, recall = compute_precision_recall(
            image, annotation, hit_annotations, mpp=0.5, tile_size=(224, 224), distance_cutoff=15000
        )
        if plot_results:
            plot_wsi_and_annotation_overlay(
                image, annotation, hit_annotations, mpp=16, tile_size=(7, 7), filename_appendage=filename
            )
        # iou = compute_iou_from_hits(image, annotation, hits)
        # iou_per_wsi[filename] = iou
        precision_recall_per_wsi[filename] = (precision, recall)
    return precision_recall_per_wsi


def perform_vector_lookup(
    train_collection_name: str,
    test_collection_name: str,
    annotated_region_overlaps: list[float] | float,
    search_radii: list[float] | float,
    limit_results: int = 16000,
    range_filter: float = 1.0,
    print_results: bool = True,
    plot_results: bool = False,
    force_recompute_annotated_vectors: bool = False,
) -> dict[str, list[float]]:
    if isinstance(annotated_region_overlaps, float):
        annotated_region_overlaps = [annotated_region_overlaps]
    if isinstance(search_radii, float):
        search_radii = [search_radii]

    data_description_annotated = load_data_description(os.environ.get("DATA_DESCRIPTION_PATH_ANNOTATED"))
    data_description_test = load_data_description(os.environ.get("DATA_DESCRIPTION_PATH_TEST"))
    connect_to_milvus(
        host=os.environ.get("MILVUS_HOST"),
        user=os.environ.get("MILVUS_USER"),
        password=os.environ.get("MILVUS_PASSWORD"),
        port=os.environ.get("MILVUS_PORT"),
        alias=os.environ.get("MILVUS_ALIAS"),
    )
    collection_train = Collection(name=train_collection_name, using="ahcore_milvus_vector_db")
    collection_test = Collection(name=test_collection_name, using="ahcore_milvus_vector_db")

    create_index(collection_train)
    create_index(collection_test)
    collection_train.load()
    collection_test.load()

    annotated_vec_querier = AnnotatedVectorQuerier(
        collection=collection_train, data_description=data_description_annotated
    )

    result_dict = {}
    for overlap in annotated_region_overlaps:
        average_annotated_vector = annotated_vec_querier.get_reference_vectors(
            overlap_threshold=overlap, force_recompute=force_recompute_annotated_vectors
        )

        for search_radius in search_radii:
            search_results = search_vectors_in_radius(
                collection_test,
                reference_vector=average_annotated_vector,
                limit_results=limit_results,
                search_radius=search_radius,
                range_filter=range_filter,
            )
            metrics_per_wsi = compute_metrics_per_wsi(
                search_results, data_description=data_description_test, plot_results=plot_results
            )
            for filename, metrics in metrics_per_wsi.items():
                if filename not in result_dict:
                    result_dict[filename] = []
                local_result_dict = {"search_radius": search_radius, "overlap": overlap, "results": metrics}
                result_dict[filename].append(local_result_dict)

            if print_results:
                print(f"----------------- Search radius: {search_radius} overlap: {overlap} -----------------")
                print(metrics_per_wsi)
                average_precision = (
                    sum([x[0] for x in metrics_per_wsi.values()]) / len(metrics_per_wsi)
                    if len(metrics_per_wsi) > 0
                    else 0
                )
                average_recall = sum([x[1] for x in metrics_per_wsi.values()]) / len(metrics_per_wsi)
                print(f"Average precision: {average_precision}")
                print(f"Average recall: {average_recall}")
                print(f"F1 score: {2 * average_precision * average_recall / (average_precision + average_recall)}")
    return result_dict


if __name__ == "__main__":
    perform_vector_lookup(
        "uni_collection_concat_train",
        "uni_collection_concat_test",
        [0.5],
        [0.67],
        print_results=True,
        plot_results=False,
        force_recompute_annotated_vectors=False,
    )
