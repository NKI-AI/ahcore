import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


class LogImagesCallback(pl.Callback):
    def __init__(self, color_map):
        super().__init__()
        self.color_map = {k: np.array(to_rgb(v)) * 255 for k, v in color_map.items()}
        self._already_seen_scanner = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if trainer.current_epoch % 10 == 0:
            scanner_name = None
            val_images = batch['image']
            path = batch['path'][0]
            if path.split('.')[-1] == 'svs':
                scanner_name = "Aperio"
            elif path.split('.')[-1] == 'mrxs':
                scanner_name = "P1000"

            if scanner_name not in self._already_seen_scanner:
                val_images_numpy = val_images.permute(0, 2, 3, 1).detach().cpu().numpy()
                val_images_numpy = (val_images_numpy - val_images_numpy.min()) / (
                            val_images_numpy.max() - val_images_numpy.min())

                val_predictions = outputs['prediction']
                val_predictions_numpy = val_predictions.permute(0, 2, 3, 1).detach().cpu().numpy()

                val_targets = batch['target']
                val_targets_numpy = val_targets.permute(0, 2, 3, 1).detach().cpu().numpy()

                self._plot_and_log(val_images_numpy, val_predictions_numpy, val_targets_numpy, pl_module, trainer.global_step,
                                   batch_idx, scanner_name)
                self._already_seen_scanner.append(scanner_name)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._already_seen_scanner = []

    def _plot_and_log(self, images_numpy, predictions_numpy, targets_numpy, pl_module, step, batch_idx, scanner_name) -> None:
        batch_size = images_numpy.shape[0]
        figure = plt.figure(figsize=(15, batch_size * 5))  # Adjust the figure size to fit the grid
        for i in range(batch_size):
            # Plot the original image
            plt.subplot(batch_size, 3, i * 3 + 1)
            plt.imshow(images_numpy[i])
            plt.axis("off")
            plt.title(f"Original Image (Scanner: {scanner_name})")

            # Plot the ground truth mask
            plt.subplot(batch_size, 3, i * 3 + 2)
            class_indices_gt = np.argmax(targets_numpy[i], axis=-1)
            colored_img_gt = apply_color_map(class_indices_gt, self.color_map)  # Apply color map to ground truth
            plt.imshow(colored_img_gt)
            plt.axis("off")

            # Plot the prediction mask
            plt.subplot(batch_size, 3, i * 3 + 3)
            class_indices_pred = np.argmax(predictions_numpy[i], axis=-1)
            colored_img_pred = apply_color_map(class_indices_pred, self.color_map)  # Apply color map to prediction
            plt.imshow(colored_img_pred)
            plt.axis("off")
            plt.tight_layout()

        artifact_file_name = f"validation_global_step{step:03d}_batch{batch_idx:03d}.png"
        pl_module.logger.experiment.log_figure(
            pl_module.logger.run_id,
            figure,
            artifact_file=artifact_file_name
        )
        plt.close()


def apply_color_map(image, color_map):
    colored_image = np.zeros((*image.shape, 3), dtype=np.uint8)
    for i in range(1, 4):  # Assuming classes are 1, 2, 3
        colored_image[image == i] = color_map[i]
    return colored_image
