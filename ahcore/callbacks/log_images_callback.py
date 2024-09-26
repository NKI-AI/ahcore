from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from matplotlib.colors import to_rgb

from ahcore.metrics.metrics import _compute_dice
from ahcore.utils.types import ScannerEnum


def get_sample_wise_dice_components(
    predictions: torch.Tensor,
    target: torch.Tensor,
    roi: torch.Tensor | None,
    num_classes: int,
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    # Apply softmax along the class dimension
    soft_predictions = F.softmax(predictions, dim=1)

    # Get the predicted class per pixel
    predictions = soft_predictions.argmax(dim=1)

    # Get the target class per pixel
    _target = target.argmax(dim=1)

    # Initialize list to store dice components for each sample
    dice_components = []

    # Loop over the batch (B samples)
    for batch_idx in range(predictions.size(0)):  # Loop through the batch dimension (B)
        batch_dice_components = []

        for class_idx in range(num_classes):  # Loop through the classes
            # Get predictions and target for the current class and sample
            curr_predictions = (predictions[batch_idx] == class_idx).int()
            curr_target = (_target[batch_idx] == class_idx).int()

            if roi is not None:
                # Apply ROI if it's provided
                curr_roi = roi[batch_idx].squeeze(0)  # Adjust ROI for current sample
                intersection = torch.sum((curr_predictions * curr_target) * curr_roi, dim=(0, 1))
                cardinality = torch.sum(curr_predictions * curr_roi, dim=(0, 1)) + torch.sum(
                    curr_target * curr_roi, dim=(0, 1)
                )
            else:
                # No ROI, just compute intersection and cardinality
                intersection = torch.sum(curr_predictions * curr_target, dim=(0, 1))
                cardinality = torch.sum(curr_predictions, dim=(0, 1)) + torch.sum(curr_target, dim=(0, 1))

            batch_dice_components.append((intersection, cardinality))

        dice_components.append(batch_dice_components)

    return dice_components


def get_sample_wise_dice(
    outputs: dict[str, Any], batch: dict[str, Any], roi: torch.Tensor, num_classes: int
) -> List[dict[int, float]]:
    dices = []
    dice_components = get_sample_wise_dice_components(outputs["prediction"], batch["target"], roi, num_classes)
    for samplewise_dice_components in dice_components:
        classwise_dices_for_a_sample = {1: 0, 2: 0, 3: 0}
        for class_idx in range(num_classes):
            if class_idx == 0:
                continue
            intersection, cardinality = samplewise_dice_components[class_idx]
            dice_score = _compute_dice(intersection, cardinality)
            classwise_dices_for_a_sample[class_idx] = dice_score.item()
        dices.append(classwise_dices_for_a_sample)
    return dices


def _extract_scanner_name(path) -> str:
    # Extract file extension
    extension = path.split(".")[-1]
    # Use the ScannerEnum to get the scanner name
    return ScannerEnum.get_scanner_name(extension)


class LogImagesCallback(pl.Callback):
    def __init__(
        self,
        color_map: dict[int, str],
        num_classes: int,
        plot_dice: bool = True,
        plot_scanner_wise: bool = False,
        plot_every_n_epochs: int = 10,
    ):
        super().__init__()
        self.color_map = {k: np.array(to_rgb(v)) * 255 for k, v in color_map.items()}
        self._num_classes = num_classes
        self._already_seen_scanner = []
        self._plot_scanner_wise = plot_scanner_wise
        self._plot_every_n_epochs = plot_every_n_epochs
        self._plot_dice = plot_dice

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if trainer.current_epoch % self._plot_every_n_epochs == 0:
            val_images = batch["image"]
            roi = batch["roi"]
            path = batch["path"][0]

            if self._plot_scanner_wise:
                scanner_name = _extract_scanner_name(path)
            else:
                scanner_name = None

            if self._plot_dice:
                dices = get_sample_wise_dice(outputs, batch, roi, self._num_classes)
            else:
                dices = None

            val_images_numpy = val_images.permute(0, 2, 3, 1).detach().cpu().numpy()
            val_images_numpy = (val_images_numpy - val_images_numpy.min()) / (
                val_images_numpy.max() - val_images_numpy.min()
            )

            val_predictions = outputs["prediction"]
            val_predictions_numpy = val_predictions.permute(0, 2, 3, 1).detach().cpu().numpy()

            val_targets = batch["target"]
            val_targets_numpy = val_targets.permute(0, 2, 3, 1).detach().cpu().numpy()

            if scanner_name not in self._already_seen_scanner:
                self._plot_and_log(
                    val_images_numpy,
                    val_predictions_numpy,
                    val_targets_numpy,
                    pl_module,
                    trainer.global_step,
                    batch_idx,
                    scanner_name,
                    dices,
                )

            if self._plot_scanner_wise and scanner_name is not None:
                self._already_seen_scanner.append(scanner_name)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._already_seen_scanner = []

    def _plot_and_log(
        self,
        images_numpy: np.ndarray,
        predictions_numpy: np.ndarray,
        targets_numpy: np.ndarray,
        pl_module: "pl.LightningModule",
        step: int,
        batch_idx: int,
        scanner_name: str,
        dices: List[dict[int, float]],
    ) -> None:
        batch_size = images_numpy.shape[0]
        figure = plt.figure(figsize=(15, batch_size * 5))
        for i in range(batch_size):
            if dices is not None:
                class_wise_dice = dices[i]
            plt.subplot(batch_size, 3, i * 3 + 1)
            plt.imshow(images_numpy[i])
            plt.axis("off")
            if scanner_name is not None:
                plt.title(f"Original Image (Scanner: {scanner_name})")

            plt.subplot(batch_size, 3, i * 3 + 2)
            class_indices_gt = np.argmax(targets_numpy[i], axis=-1)
            colored_img_gt = apply_color_map(class_indices_gt, self.color_map, self._num_classes)
            plt.imshow(colored_img_gt)
            plt.axis("off")

            plt.subplot(batch_size, 3, i * 3 + 3)
            class_indices_pred = np.argmax(predictions_numpy[i], axis=-1)
            colored_img_pred = apply_color_map(class_indices_pred, self.color_map, self._num_classes)
            if dices is not None:
                plt.title(f"Dice: {class_wise_dice[1]:.2f} {class_wise_dice[2]:.2f} {class_wise_dice[3]:.2f}")
            plt.imshow(colored_img_pred)
            plt.axis("off")
            plt.tight_layout()

        logger = pl_module.logger

        if hasattr(logger.experiment, "log_figure"):  # MLFlow logger case
            artifact_file_name = f"validation_global_step{step:03d}_batch{batch_idx:03d}.png"
            logger.experiment.log_figure(logger.run_id, figure, artifact_file=artifact_file_name)
        elif hasattr(logger.experiment, "add_figure"):  # TensorBoard logger case
            logger.experiment.add_figure(f"validation_step_{step}_batch_{batch_idx}", figure, global_step=step)
        else:
            # If another logger is being used, raise a warning or add additional logic
            raise NotImplementedError(f"Logging method for logger {type(logger).__name__} not implemented.")

        plt.close()


def apply_color_map(image, color_map, num_classes) -> np.ndarray:
    colored_image = np.zeros((*image.shape, 3), dtype=np.uint8)
    for i in range(1, num_classes):
        colored_image[image == i] = color_map[i]
    return colored_image
