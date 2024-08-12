import os
from logging import getLogger

import dotenv
import hydra
from pymilvus import RRFRanker, WeightedRanker

from ahcore.utils.vector_database.annotated_vector_querier import AnnotatedVectorQuerier
from ahcore.utils.vector_database.utils import (
    compute_global_metrics,
    compute_metrics_per_wsi,
    connect_to_milvus,
    load_data_description,
)
from ahcore.utils.vector_database.vector_searcher import VectorSearcher

dotenv.load_dotenv(override=True)
logger = getLogger(__name__)


def perform_vector_lookup_single(
    data_setup: dict[str, str],
    annotated_region_overlap: float,
    search_radius: list[float] | float,
    limit_results: int = 16000,
    range_filter: float = 1.0,
    print_results: bool = True,
    plot_results: bool = False,
    force_recompute_annotated_vectors: bool = False,
) -> dict[str, list[float]]:
    data_description_annotated = load_data_description(data_setup["data_description_path_annotated"])
    data_description_test = load_data_description(data_setup["data_description_path_test"])
    train_collection_name = data_setup["train_collection_name"]
    test_collection_name = data_setup["test_collection_name"]

    annotated_vec_querier = AnnotatedVectorQuerier(
        collection_name=train_collection_name, data_description=data_description_annotated
    )

    vec_searcher = VectorSearcher(
        collection_name=test_collection_name,
        data_description=data_description_test,
        alias=os.environ.get("MILVUS_ALIAS"),
        use_partitions=False,
    )

    logger.info(f"Performing annotatated vector lookup with overlap: {annotated_region_overlap}")
    average_annotated_vector = annotated_vec_querier.get_reference_vectors(
        overlap_threshold=annotated_region_overlap,
        reduce_method="mean",
        force_recompute=force_recompute_annotated_vectors,
    )

    search_param = {
        "metric_type": "COSINE",
        "params": {
            "radius": search_radius,
            "range_filter": range_filter,
        },
    }
    logger.info(f"Performing vector search with search radius: {search_radius}")
    search_results = vec_searcher.search_single_vector(
        reference_vector=average_annotated_vector,
        limit_results=limit_results,
        search_param=search_param,
    )

    logger.info("Computing metrics")
    metrics_per_wsi = compute_metrics_per_wsi(
        search_results, data_description=data_description_test, plot_results=plot_results
    )
    precision, recall, f1_score = compute_global_metrics(metrics_per_wsi)

    if print_results:
        print(f"----------------- Search radius: {search_radius} overlap: {annotated_region_overlap} -----------------")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1_score}")

    return f1_score


def perform_vector_lookup_multi(
    data_setup: dict[str, str],
    annotated_region_overlap: float,
    reduce_method: str,
    search_radius: list[float] | float,
    ranker: WeightedRanker | RRFRanker | None = None,
    limit_results: int = 16000,
    range_filter: float = 1.0,
    min_count: int = 1,
    print_results: bool = True,
    plot_results: bool = False,
    force_recompute_annotated_vectors: bool = False,
) -> dict[str, list[float]]:
    data_description_annotated = load_data_description(data_setup["data_description_path_annotated"])
    data_description_test = load_data_description(data_setup["data_description_path_test"])
    train_collection_name = data_setup["train_collection_name"]
    test_collection_name = data_setup["test_collection_name"]

    annotated_vec_querier = AnnotatedVectorQuerier(
        collection_name=train_collection_name, data_description=data_description_annotated
    )

    vec_searcher = VectorSearcher(
        collection_name=test_collection_name,
        data_description=data_description_test,
        alias=os.environ.get("MILVUS_ALIAS"),
        use_partitions=False,
    )

    logger.info(f"Performing annotatated vector lookup with overlap: {annotated_region_overlap}")
    annotated_vectors = annotated_vec_querier.get_reference_vectors(
        overlap_threshold=annotated_region_overlap,
        reduce_method=reduce_method,
        force_recompute=force_recompute_annotated_vectors,
    )

    search_param = {
        "metric_type": "COSINE",
        "params": {
            "radius": search_radius,
            "range_filter": range_filter,
        },
    }
    logger.info(f"Performing vector search with search radius: {search_radius}")
    if ranker is None:
        search_results = vec_searcher.search_multi_vector(
            reference_vectors=annotated_vectors,
            limit_results=limit_results,
            search_param=search_param,
            min_count=min_count,
        )
    else:
        search_results = vec_searcher.search_multi_vector_rerank(
            reference_vectors=annotated_vectors, ranker=ranker, limit_results=limit_results, search_param=search_param
        )

    logger.info("Computing metrics")
    metrics_per_wsi = compute_metrics_per_wsi(
        search_results, data_description=data_description_test, plot_results=plot_results
    )
    precision, recall, f1_score = compute_global_metrics(metrics_per_wsi)

    if print_results:
        print(f"----------------- Search radius: {search_radius} overlap: {annotated_region_overlap} -----------------")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1_score}")

    return f1_score


@hydra.main(
    config_path="config",
    config_name="main.yaml",
    version_base="1.3",
)
def main(cfg):
    connect_to_milvus(
        host=os.environ.get("MILVUS_HOST"),
        user=os.environ.get("MILVUS_USER"),
        password=os.environ.get("MILVUS_PASSWORD"),
        port=os.environ.get("MILVUS_PORT"),
        alias=os.environ.get("MILVUS_ALIAS"),
    )
    f1_score = hydra.utils.call(cfg.vector_lookup, data_setup=cfg.data_setup, _convert_="partial")
    return f1_score


if __name__ == "__main__":
    main()
