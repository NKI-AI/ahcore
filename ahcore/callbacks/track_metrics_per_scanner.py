import pytorch_lightning as pl

from ahcore.metrics import TileMetric
from ahcore.utils.callbacks import AhCoreLogger
from ahcore.utils.types import ScannerVendors


class TrackTileMetricsPerScanner(pl.Callback):
    """
    This callback is used to track several `TileMetric` from ahcore per scanner.
    The callback works on certain assumptions:
    - You want to track metrics corresponding to each class in the `index_map`
    - Each metric is tracked per scanner
    """
    def __init__(self, metrics: list[TileMetric], index_map: dict[str, int]):
        super().__init__()
        self.metrics = metrics
        self.index_map = index_map
        self._metrics_per_scanner = {
            scanner.scanner_name: {f"{metric.name}/{class_name}": 0.0 for class_name in self.index_map.keys() for metric in self.metrics}
            for scanner in ScannerVendors
        }
        self._batch_count_per_scanner = {scanner.scanner_name: 0 for scanner in ScannerVendors}
        self._logger: AhCoreLogger | None = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if self._logger is None:
            self._logger = AhCoreLogger(pl_module.logger)

        path = batch["path"][0]
        file_extension = path.split(".")[-1]
        scanner_name = ScannerVendors.get_vendor_name(file_extension)

        prediction = outputs["prediction"]
        target = batch["target"]
        roi = batch.get("roi", None)

        for metric in self.metrics:
            batch_metrics = metric(prediction, target, roi)
            for class_name, class_index in self.index_map.items():
                metric_key = f"{metric.name}/{class_name}"
                self._metrics_per_scanner[scanner_name][metric_key] += batch_metrics[metric_key].item()

        self._batch_count_per_scanner[scanner_name] += 1

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for scanner_name, metrics in self._metrics_per_scanner.items():
            batch_count = self._batch_count_per_scanner[scanner_name]
            if batch_count > 0:
                averaged_metrics = {f"{scanner_name}/{key}": value / batch_count for key, value in metrics.items()}
                self._logger.log_metrics(averaged_metrics, step=trainer.global_step)

        self._metrics_per_scanner = {
            scanner.scanner_name: {f"{metric.name}/{class_name}": 0.0 for class_name in self.index_map.keys() for metric in self.metrics}
            for scanner in ScannerVendors
        }
        self._batch_count_per_scanner = {scanner.scanner_name: 0 for scanner in ScannerVendors}
