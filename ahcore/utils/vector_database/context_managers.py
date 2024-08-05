from collections import defaultdict
from contextlib import contextmanager
from abc import ABC, abstractmethod
from pymilvus import Collection


import threading


class MilvusResourceManager(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.collection: Collection

    @abstractmethod
    def load_resource(self, resource_identifiers: list[str] | None = None):
        pass

    @abstractmethod
    def release_resource(self, resource_identifiers: list[str] | None = None):
        pass

    @abstractmethod
    def manage_resource(self, resource_identifiers: list[str] | None = None):
        pass


class ManagedMilvusCollection(MilvusResourceManager):
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
    
    def load_resource(self, resource_identifiers: list[str] | None = None):
        with self.access_lock:
            if self.ref_count == 0:
                self.collection.load()
            self.ref_count += 1
    
    def release_resource(self, resource_identifiers: list[str] | None = None):
        with self.access_lock:
            self.ref_count -= 1
            if self.ref_count == 0:
                self.collection.release()

    def manage_resource(self, resource_identifiers: list[str] | None = None):
        if resource_identifiers is not None:
            raise ValueError("ManagedMilvusCollection only manages complete collections.")
        
        self.load_resource(resource_identifiers)
        try:
            yield self.collection
        finally:
            self.release_resource(resource_identifiers)


class ManagedMilvusPartitions(MilvusResourceManager):
    """Singleton to manage reference counting of partitions in a Milvus collection for load/release.
       Used when the complete collection is too large to load at once, and partitions are loaded inst."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, collection_name, alias="default"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.alias = alias
                cls._instance.collection = Collection(name=collection_name, using=cls._instance.alias)
                cls._instance.partition_ref_counts = defaultdict(int)
                cls._instance.access_lock = threading.Lock()
            return cls._instance

    @contextmanager
    def manage_resource(self, resource_identifiers: list[str] | None = None):
        """Context manager for loading and releasing partitions."""
        if resource_identifiers is None:
            raise ValueError("ManagedMilvusPartitions requires partition identifiers.")
        self.load_resource(resource_identifiers)
        try:
            yield self.collection
        finally:
            self.release_resource(resource_identifiers)

    def load_resource(self, resource_identifiers: list[str] | None = None):
        """Loads the specified partitions, managing reference counts."""
        with self.access_lock:
            for name in resource_identifiers:
                if self.partition_ref_counts[name] == 0:
                    self.collection.load(partitions=[name])
                self.partition_ref_counts[name] += 1

    def release_resource(self, resource_partitions: list[str] | None = None):
        """Releases the specified partitions if no more references exist."""
        with self.access_lock:
            for name in resource_partitions:
                if self.partition_ref_counts[name] > 0:
                    self.partition_ref_counts[name] -= 1
                    if self.partition_ref_counts[name] == 0:
                        self.collection.release(partitions=[name])