from logging import getLogger
from typing import Any

import dotenv
import numpy as np
from pymilvus import AnnSearchRequest, Collection, RRFRanker, WeightedRanker

from ahcore.utils.data import DataDescription
from ahcore.utils.vector_database.context_managers import ManagedMilvusCollection, ManagedMilvusPartitions

log = getLogger(__name__)
dotenv.load_dotenv(override=True)


class VectorSearcher:
    # TODO: Async search
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
                "params": {},
            }
        return search_param

    def search_single_vector(
        self,
        reference_vector: list[float],
        limit_results: int = 10,
        partitions: list[str] | None = None,
        search_param: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search with a single reference vector in the Milvus collection.
        """
        param = self._setup_search_param(search_param)

        with self._manager.manage_resource(partitions) as collection:
            results = collection.search(
                data=[reference_vector],
                anns_field=self._vector_entry_name,
                param=param,
                limit=limit_results,
                partition_names=partitions,
            )

        return results

    def search_multi_vector(
        self,
        reference_vectors: list[list[float]],
        ranker: WeightedRanker | RRFRanker,
        limit_results: int = 10,
        partitions: list[str] | None = None,
        search_param: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search with multiple reference vectors in the Milvus collection.
        Results of the multiple vectors are subsequently re-ranked using the ranker.
        """
        base_param = self._setup_search_param(search_param)

        # prepare AnnSearches
        searches = []
        for vector in reference_vectors:
            param = base_param.copy()
            searches.append(
                AnnSearchRequest(data=[vector], anns_field=self._vector_entry_name, param=param, limit=limit_results)
            )

        with self._manager.manage_resource(partitions) as collection:
            results = collection.hybrid_search(reqs=searches, rerank=ranker, partition_names=partitions)

        return results

    @staticmethod
    def _extract_embeddings(entries_dict: list[dict[str, Any]]) -> np.ndarray:
        embeddings = [entry["embedding"] for entries in entries_dict.values() for entry in entries]
        embeddings_np = np.array(embeddings)
        return embeddings_np
