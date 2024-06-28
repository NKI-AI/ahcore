from logging import getLogger
from typing import Any, Callable, Dict

from pymilvus import Collection, MilvusException, connections
from pytorch_lightning import Callback, LightningModule, Trainer

from ahcore.utils.vector_database.factories import (
    CollectionType,
    MilvusCollectionFactory,
    PrepareDataFactory,
    PrepareDataTypeDebug,
    PrepareDataTypePathCLS,
)

logger = getLogger(__name__)


class MilvusVectorDBCallback(Callback):
    def __init__(
        self,
        host: str,
        port: str,
        user: str,
        password: str,
        collection_name: str,
        collection_type: str,
        embedding_dim: int,
        database: str = "default",
    ):
        """
        Callback to send outputs to Milvus vector database.

        Parameters
        ----------
        host : str
            The host of the Milvus server.
        port : str
            The port of the Milvus server.
        user : str
            The user to connect to the Milvus server.
        password : str
            The password for the user.
        collection_name : str
            The name of the collection to insert data into.
        collection_type : str
            The type of the collection to insert data into, this will invoke a CollectionType, with specified types
            and data preparation. E.g. "path_fomo_cls" for a pathology fomo with Image metadata and CLS embeddings.
        embedding_dim: int
            The output dimension of your embedding model
        database : str
            The database type to connect to. Default is "default".
        """
        self.host = host
        self.port = int(port)
        self.user = user
        self.database = database
        self._password = password
        self._connection_alias = "ahcore_milvus_vector_db"
        connections.connect(
            host=self.host, port=self.port, user=self.user, password=self._password, alias=self._connection_alias
        )

        self._collection_name = collection_name
        self._collection_type = CollectionType[collection_type.upper()]
        self._collection: Collection | None = None
        self._prepare_data: Callable[
            [Dict[str, Any], Dict[str, Any]], PrepareDataTypePathCLS | PrepareDataTypeDebug
        ] | None = None

        self.embedding_dim = embedding_dim

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        try:
            self._collection = Collection(self._collection_name, using=self._connection_alias)
        except MilvusException:
            logger.info(f"Collection {self._collection_name} does not exist. Creating it now.")
            factory = MilvusCollectionFactory(embedding_dim=self.embedding_dim)
            schema = factory.get_schema(self._collection_type)
            self._collection = Collection(self._collection_name, schema, using=self._connection_alias)

        prepare_data_factory: PrepareDataFactory = PrepareDataFactory()
        self._prepare_data = prepare_data_factory.get_preparer(self._collection_type)

    @property
    def collection(self) -> Collection:
        return self._collection

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # for mypy
        assert self._prepare_data, "prepare_data should be set in the setup method."
        assert self._collection, "collection should be set in the setup method."

        prepared_data = list(self._prepare_data(outputs, batch))  # convert to list for milvus

        # Insert data into Milvus collection
        try:
            self._collection.insert(prepared_data)
        except MilvusException as e:
            logger.error(f"Error inserting data into Milvus collection: {e}")

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.collection.flush()  # this is last flush, milvus auto-flushes whenever reaching a certain cache size
        connections.disconnect(self._connection_alias)
