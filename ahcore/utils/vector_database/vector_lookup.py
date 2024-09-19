import os
from logging import getLogger
from multiprocessing import Pool
from typing import Any

import dotenv
import hydra
from pymilvus import RRFRanker, WeightedRanker

from ahcore.utils.vector_database.annotated_vector_querier import AnnotatedVectorQuerier
from ahcore.utils.vector_database.metrics import Metrics
from ahcore.utils.vector_database.utils import connect_to_milvus, load_data_description
from ahcore.utils.vector_database.vector_searcher import VectorSearcher

dotenv.load_dotenv(override=True)
logger = getLogger(__name__)


def perform_vector_lookup(
    data_setup: dict[str, str],
    annotated_region_overlap: float,
    reduce_method: str,
    search_radius: list[float] | float,
    target_filenames: list[str] | str,
    annotated_search_kwargs: dict[str, Any] = {},
    ranker: WeightedRanker | RRFRanker | None = None,
    force_recompute_annotated_vectors: bool = False,
    limit_results: int = 16000,
    range_filter: float = 1.0,
    min_count: int = 1,
    print_results: bool = True,
    plot_results: bool = False,
    num_processes: int = 0,
) -> dict[str, list[float]] | None:
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
        **annotated_search_kwargs,
    )

    search_param = {
        "metric_type": "COSINE",
        "params": {
            "radius": search_radius,
            "range_filter": range_filter,
        },
    }

    logger.info(f"Performing vector search with search radius: {search_radius}")
    if isinstance(target_filenames, str):
        target_filenames = [target_filenames]

    all_search_results = []

    if num_processes > 1:
        num_processes = min(num_processes, len(target_filenames))
        logger.info(f"Using {num_processes} processes")

        # Prepare arguments for each process
        args_list = [
            (
                annotated_vectors,
                limit_results,
                search_param,
                min_count,
                filename,
                ranker,
                vec_searcher,
            )
            for filename in target_filenames
        ]

        with Pool(processes=num_processes) as pool:
            results_list = pool.map(process_single_filename, args_list)

        # Collect all results
        for search_results in results_list:
            all_search_results.extend(search_results)
    else:
        # Process each filename sequentially
        chunk_size = 2
        for i in range(0, len(target_filenames), chunk_size):
            filenames = target_filenames[i : i + chunk_size]
            logger.info(f"Processing filenames: {filenames}")

            if ranker is None:
                search_results = vec_searcher.search_vectors(
                    reference_vectors=annotated_vectors,
                    limit_results=limit_results,
                    search_param=search_param,
                    min_count=min_count,
                    filenames=filenames,
                )
            else:
                search_results = vec_searcher.search_vectors_rerank(
                    reference_vectors=annotated_vectors,
                    ranker=ranker,
                    limit_results=limit_results,
                    search_param=search_param,
                    filenames=filenames,
                )

            # Collect the results
            all_search_results.extend(search_results)

    logger.info("Computing metrics")
    metrics = Metrics(data_description=data_description_test, metrics=["iou", "precision", "recall", "f1"])
    metrics_per_wsi, global_metrics = metrics.compute_metrics(all_search_results, plot_results=plot_results)

    if print_results:
        print(f"----------------- Search radius: {search_radius} overlap: {annotated_region_overlap} -----------------")
        print(f"Global metrics: {global_metrics}")

    return global_metrics.get("f1", None)


def process_single_filename(args):
    (
        annotated_vectors,
        limit_results,
        search_param,
        min_count,
        filename,
        ranker,
        vec_searcher,
    ) = args

    logger.info(f"Processing filename: {filename}")

    if ranker is None:
        search_results = vec_searcher.search_vectors(
            reference_vectors=annotated_vectors,
            limit_results=limit_results,
            search_param=search_param,
            min_count=min_count,
            filenames=filename,
        )
    else:
        search_results = vec_searcher.search_vectors_rerank(
            reference_vectors=annotated_vectors,
            ranker=ranker,
            limit_results=limit_results,
            search_param=search_param,
            filenames=filename,
        )

    return search_results


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
