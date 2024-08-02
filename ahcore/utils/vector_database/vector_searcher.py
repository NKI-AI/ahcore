import threading
from contextlib import contextmanager
from logging import getLogger
from typing import Any, Generator

import dotenv
from pymilvus import Collection

from ahcore.utils.data import DataDescription

log = getLogger(__name__)
dotenv.load_dotenv(override=True)


class ManagedMilvusCollection:
    """Singleton to manage reference counting of a Milvus collection for load/release."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, collection_name, alias="default"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.alias = alias
                cls._instance.collection = Collection(name=collection_name, using=cls._instance.alias)
                cls._instance.ref_count = 0
                cls._instance.access_lock = threading.Lock()
            return cls._instance

    def __enter__(self):
        with self.access_lock:
            if self.ref_count == 0:
                self.collection.load()
            self.ref_count += 1
        return self.collection

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.access_lock:
            self.ref_count -= 1
            if self.ref_count == 0:
                self.collection.release()


class VectorSearcher:
    # TODO: Async search
    def __init__(self, collection_name: str, data_description: DataDescription, alias: str = "default"):
        self.collection_name = collection_name
        self.alias = alias
        self.data_description = data_description

    @contextmanager
    def manage_collection(self, partition_names: list[str] | None = None) -> Generator[Collection, None, None]:
        try:
            self.collection.load(partition_names=partition_names)
            yield self.collection
        finally:
            self.collection.release(partition_names=partition_names)

    def _setup_search_param(self, search_param: dict[str, Any] | None = None) -> dict[str, Any]:
        if search_param is None:
            search_param = {
                "metric_type": "COSINE",
                "params": {},
            }
        return search_param

    def _search_single_vector(
        self,
        reference_vector: list[float],
        limit_results: int = 10,
        partitions: list[str] | None = None,
        search_param: dict[str, Any] | None = None,
    ) -> list[list[str, Any]]:
        """
        Search for a single vector in the Milvus collection.
        """
        param = self._setup_search_param(search_param)

        # Execute the query
        results = self.collection.search(
            data=[reference_vector],
            anns_field="embedding",
            param=param,
            limit=limit_results,
            partition_names=partitions,
        )

        return results
