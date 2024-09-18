import pytorch_lightning as pl
from ahcore.metrics import TileMetric


class TrackMetricsPerScanner(pl.Callback):
    def __init__(self, metrics: TileMetric):
        super().__init__()
        self.metrics = metrics[0]

        # Initialize accumulators for Aperio and P1000 metrics
        self._aperio_metrics = {'dice/background': 0.0, 'dice/stroma': 0.0, 'dice/tumor': 0.0, 'dice/ignore': 0.0}
        self._p1000_metrics = {'dice/background': 0.0, 'dice/stroma': 0.0, 'dice/tumor': 0.0, 'dice/ignore': 0.0}

        # Track the number of batches for each scanner
        self._aperio_batch_count = 0
        self._p1000_batch_count = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        # Determine the scanner based on the file extension
        scanner_name = None
        path = batch['path'][0]
        if path.split('.')[-1] == 'svs':
            scanner_name = "Aperio"
        elif path.split('.')[-1] == 'mrxs':
            scanner_name = "P1000"

        prediction = outputs['prediction']
        target = batch['target']
        roi = batch.get('roi', None)

        # Get the metrics for the current batch
        batch_metrics = self.metrics(prediction, target, roi)

        # Accumulate metrics based on the scanner
        if scanner_name == "Aperio":
            self._aperio_metrics['dice/background'] += batch_metrics['dice/background'].item()
            self._aperio_metrics['dice/stroma'] += batch_metrics['dice/stroma'].item()
            self._aperio_metrics['dice/tumor'] += batch_metrics['dice/tumor'].item()
            self._aperio_metrics['dice/ignore'] += batch_metrics['dice/ignore'].item()
            self._aperio_batch_count += 1

        elif scanner_name == "P1000":
            self._p1000_metrics['dice/background'] += batch_metrics['dice/background'].item()
            self._p1000_metrics['dice/stroma'] += batch_metrics['dice/stroma'].item()
            self._p1000_metrics['dice/tumor'] += batch_metrics['dice/tumor'].item()
            self._p1000_metrics['dice/ignore'] += batch_metrics['dice/ignore'].item()
            self._p1000_batch_count += 1

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Compute the average metrics for Aperio
        if self._aperio_batch_count > 0:
            averaged_aperio_metrics = {f"Aperio/{key}": value / self._aperio_batch_count for key, value in
                                       self._aperio_metrics.items()}
            trainer.logger.log_metrics(averaged_aperio_metrics, step=trainer.global_step)

        # Compute the average metrics for P1000
        if self._p1000_batch_count > 0:
            averaged_p1000_metrics = {f"P1000/{key}": value / self._p1000_batch_count for key, value in
                                      self._p1000_metrics.items()}
            trainer.logger.log_metrics(averaged_p1000_metrics, step=trainer.global_step)

        self._aperio_metrics = {'dice/background': 0.0, 'dice/stroma': 0.0, 'dice/tumor': 0.0, 'dice/ignore': 0.0}
        self._p1000_metrics = {'dice/background': 0.0, 'dice/stroma': 0.0, 'dice/tumor': 0.0, 'dice/ignore': 0.0}
        self._aperio_batch_count = 0
        self._p1000_batch_count = 0
