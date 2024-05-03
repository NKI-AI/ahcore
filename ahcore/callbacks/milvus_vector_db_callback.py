from logging import getLogger
from typing import Any, Callable, Dict

import torch
from pymilvus import Collection, MilvusException, connections
from pytorch_lightning import Callback, LightningModule, Trainer

from ahcore.utils.vector_database.factories import (
    CollectionType,
    MilvusCollectionFactory,
    PrepareDataFactory,
    PrepareDataType,
)

logger = getLogger(__name__)


class MilvusVectorDBCallback(Callback):
    def __init__(self, host: str, port: int, user: str, password: str, collection_name: str, database: str = "default"):
        self.host = host
        self.port = port
        self.user = user
        self._password = password
        self.database = database
        self._connection = connections.connect(host=self.host, port=self.port, user=self.user, password=self._password)

        self._collection_name = collection_name
        self._collection_type = CollectionType[collection_name.upper()]
        self._collection: Collection | None = None
        self._prepare_data: Callable[[torch.Tensor, Dict[str, Any]], PrepareDataType] | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        try:
            self._collection = Collection(self._collection_name)
        except MilvusException:
            logger.info(f"Collection {self._collection_name} does not exist. Creating it now.")
            factory = MilvusCollectionFactory()
            schema = factory.get_schema(self._collection_type)
            self._collection = Collection(self._collection_name, schema)

        prepare_data_factory: PrepareDataFactory = PrepareDataFactory()
        self._prepare_data = prepare_data_factory.get_preparer(self._collection_type)

    @property
    def collection(self) -> Collection:
        return self._collection

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # for mypy
        assert self._prepare_data, "prepare_data should be set in the setup method."
        assert self._collection, "collection should be set in the setup method."

        prepared_data = self._prepare_data(outputs, batch)

        # Insert data into Milvus collection
        try:
            self._collection.insert(prepared_data)
        except MilvusException as e:
            logger.error(f"Error inserting data into Milvus collection: {e}")

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # TODO: keep track if anything is still to be sent to the database before closing the connection
        self._connection.close()
