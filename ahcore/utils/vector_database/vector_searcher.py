from logging import getLogger
from typing import Any

import dotenv
import numpy as np
from pymilvus import AnnSearchRequest, Collection, RRFRanker, WeightedRanker
from pymilvus.client.abstract import SearchResult

from ahcore.utils.data import DataDescription
from ahcore.utils.vector_database.context_managers import ManagedMilvusCollection, ManagedMilvusPartitions

log = getLogger(__name__)
dotenv.load_dotenv(override=True)


class VectorSearcher:
    # TODO: Async search
    # TODO: update types or convert search result to dicts
    def __init__(
        self,
        collection_name: str,
        data_description: DataDescription,
        use_partitions: bool,
        index_param: dict[str, Any] | None = None,
        alias: str = "default",
    ):
        self.collection_name = collection_name
        self.alias = alias
        self.data_description = data_description
        self._vector_entry_name = "embedding"  # The vector entry in the Milvus collection

        if use_partitions:
            self._manager = ManagedMilvusPartitions(self.collection_name, alias=self.alias)
        else:
            self._manager = ManagedMilvusCollection(self.collection_name, alias=self.alias)

        self._setup_index(collection_name, index_param)

    def _setup_index(self, collection_name: str, index_params: dict[str, Any] | None = None) -> None:
        if index_params is None:
            index_params = {
                "index_type": "FLAT",
                "metric_type": "COSINE",
                "params": {},
            }
        Collection(name=collection_name, using=self.alias).create_index(
            field_name=self._vector_entry_name, index_params=index_params
        )

    def _setup_search_param(self, search_param: dict[str, Any] | None = None) -> dict[str, Any]:
        if search_param is None:
            search_param = {
                "metric_type": "COSINE",
                "params": {
                    "radius": 0.5,  # Radius of the search circle
                    "range_filter": 1.0,  # Range filter to filter out vectors that are not within the search circle
                },
            }
        return search_param

    def _create_filename_expr(self, filenames: list[str] | str | None = None) -> str | None:
        if filenames is not None:
            if isinstance(filenames, str):
                filenames = [filenames]
            quoted_filenames = ", ".join([f"'{filename}'" for filename in filenames])
            expr = f"filename in [{quoted_filenames}]"
        else:
            expr = None
        return expr

    @staticmethod
    def _extract_unique_entries(search_results: SearchResult) -> tuple[list[dict[str, Any]], dict[str, int]]:
        unique_ids = set()
        unique_entries = []
        id_counts = {}

        for sub_search_results in search_results:
            for hit in sub_search_results:
                if hit.id in id_counts:
                    id_counts[hit.id] += 1
                else:
                    id_counts[hit.id] = 1

                if hit.id not in unique_ids:
                    unique_ids.add(hit.id)
                    unique_entries.append(hit)

        return unique_entries, id_counts

    def search_vectors(
        self,
        reference_vectors: list[list[float]] | list[float],
        min_count: int = 1,
        limit_results: int = 10,
        partitions: list[str] | None = None,
        search_param: dict[str, Any] | None = None,
        filenames: list[str] | str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search with multiple reference vectors in the Milvus collection.
        """
        if isinstance(reference_vectors[0], (int, float)):
            reference_vectors = [reference_vectors]

        param = self._setup_search_param(search_param)
        expr = self._create_filename_expr(filenames)

        with self._manager.manage_resource(partitions) as collection:
            results = collection.search(
                data=reference_vectors,
                anns_field=self._vector_entry_name,
                param=param,
                limit=limit_results,
                partition_names=partitions,
                expr=expr,
                output_fields=["*"],
            )

        unique_entries, counts = self._extract_unique_entries(results)
        unique_entries = [entry for entry in unique_entries if counts[entry.id] >= min_count]
        log.info(f"Search completed with {len(unique_entries)} results.")
        return unique_entries

    def search_vectors_rerank(
        self,
        reference_vectors: list[list[float]] | list[float],
        ranker: WeightedRanker | RRFRanker,
        limit_results: int = 10,
        partitions: list[str] | None = None,
        search_param: dict[str, Any] | None = None,
        filenames: list[str] | str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search with multiple reference vectors in the Milvus collection.
        Results of the multiple searches are subsequently re-ranked using the ranker.
        """
        if isinstance(reference_vectors[0], (int, float)):
            reference_vectors = [reference_vectors]

        base_param = self._setup_search_param(search_param)
        expr = self._create_filename_expr(filenames)

        # prepare AnnSearches
        searches = []
        for vector in reference_vectors:
            param = base_param.copy()
            searches.append(
                AnnSearchRequest(
                    data=[vector], anns_field=self._vector_entry_name, param=param, limit=limit_results, expr=expr
                )
            )  # Note that these annsearches can also have an additional 'expr' for other fields

        with self._manager.manage_resource(partitions) as collection:
            results = collection.hybrid_search(
                reqs=searches, rerank=ranker, partition_names=partitions, limit=limit_results, output_fields=["*"]
            )
            log.info(f"Search completed with {len(results[0])} results.")

        return results

    def search_multi_vector_annreqs(
        self,
        searches: list[AnnSearchRequest],
        ranker: WeightedRanker | RRFRanker,
        partitions: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search with multiple AnnSearchRequests in the Milvus collection. Allows for individual search params
        per reference vector.
        Results of the multiple searches are subsequently re-ranked using the ranker.
        """
        with self._manager.manage_resource(partitions) as collection:
            results = collection.hybrid_search(reqs=searches, rerank=ranker, partition_names=partitions)

        return results

    @staticmethod
    def _extract_embeddings(entries_dict: list[dict[str, Any]]) -> np.ndarray:
        embeddings = [entry["embedding"] for entries in entries_dict.values() for entry in entries]
        embeddings_np = np.array(embeddings)
        return embeddings_np
