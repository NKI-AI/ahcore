# encoding: utf-8
"""
Loss factory

"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFactory(nn.Module):
    """Loss factory to construct the total loss."""

    def __init__(
        self,
        losses: list,
        weights: list[Union[torch.Tensor, float]] | None = None,
        class_proportions: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        losses : list
            List of losses which are functions which accept `(input, target, roi, weight)`. The weight will be
            applied per class.
        weights : list
            List of length `losses`. The weights weight the total contribution so `weight_0 * loss_0_val + ...` will
            be the resulting loss.
        class_proportions
        """
        super().__init__()
        if weights is None:
            weights = [1.0] * len(losses)

        _weights = [torch.Tensor([_]) for _ in weights]
        self._weights: list[torch.Tensor] = _weights

        self._losses = []
        for loss in losses:
            self._losses += list(loss.values())

        if class_proportions is not None:
            _class_weights = 1 / class_proportions
            _class_weights[_class_weights.isnan()] = 0.0
            _class_weights = _class_weights / _class_weights.max()
            self._class_weights = _class_weights
        else:
            self._class_weights = None

    def forward(self, input: torch.Tensor, target: torch.Tensor, roi: torch.Tensor | None = None):
        total_loss = sum(
            [
                weight.to(input.device) * curr_loss(input, target, roi, weight=self._class_weights)
                for weight, curr_loss in zip(self._weights, self._losses)
            ]
        )
        return total_loss


def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    roi: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
    topk: float | None = None,
    label_smoothing: float = 0.0,
    limit: float | None = None,
) -> torch.Tensor:
    """
    Compute a ROI weighted cross entropy function. The resulting output is a per-sample cross entropy.

    Parameters
    ----------
    input : torch.Tensor
        Input of shape `(N, C, H, W)`.
    target : torch.Tensor
        One-hot encoded target of shape `(N, C, H, W)`.
    roi : torch.Tensor
        ROI of shape `(N, 1, H, W)`
    weight : torch.Tensor, optional
        Per class weight
    ignore_index : int, optional
        Specifies a target value that is ignored and does not contribute to the input gradient.
    topk : float, optional
        Apply top-k in the loss
    label_smoothing : float, optional
        Float in [0, 1]. Amount of smoothing.
        `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__.
        Default: :math:`0.0`.
    limit : float, optional
        If set this will be the value the cross entropy is clipped (from below). This has to be a negative value.

    Returns
    -------
    torch.Tensor
        Output as a torch.Tensor float
    """
    corrected_roi = None
    if limit is not None:
        if limit >= 0:
            raise ValueError(f"Limit has to be a negative value. Got {limit}")

    if topk is not None:
        if topk <= 0 or topk >= 1.0:
            raise ValueError(f"topk value needs to be between 0 and 1. Got {topk}.")

    if roi is not None:
        # Correct the ROI to remove any unannotated regions (see github issue #11)
        background = (target[:, 0, :, :] == 1).int()
        corrected_roi = roi.squeeze(1) - background
        # Clip the values below 0 after the subtraction.
        corrected_roi[corrected_roi < 0] = 0
        roi_sum = corrected_roi.sum() / input.shape[0]
        if roi_sum == 0:
            return torch.Tensor([0.0]).to(input.device)
    else:
        roi_sum = torch.Tensor([np.prod(tuple(input.shape)[2:])]).to(input.device)

    if ignore_index is None:
        ignore_index = -100

    # compute cross_entropy pixel by pixel
    _cross_entropy = F.cross_entropy(
        input,
        target.argmax(dim=1),
        ignore_index=ignore_index,
        weight=None if weight is None else weight.to(input.device),
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if limit is not None:
        _cross_entropy = torch.clip(_cross_entropy, limit, None)

    if corrected_roi is not None:
        _cross_entropy = corrected_roi * _cross_entropy

    if topk is None:
        return _cross_entropy.sum(dim=(1, 2)) / roi_sum

    k = int(round(float(roi_sum.cpu()) * topk))
    return torch.topk(_cross_entropy.view(input.shape[0], -1), k).values.sum(dim=1) / (roi_sum * topk)


def soft_dice(
    input: torch.Tensor,
    target: torch.Tensor,
    roi: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    ignore_index: int | None = None,
    eps: float = 1e-17,
) -> torch.Tensor:
    """Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    The shapes of input and target need to be :math:`(N, C, H, W)` where :math:`C` = number of classes.


    Parameters
    ----------
    input : torch.Tensor
        Input of shape `(N, C, H, W)`.
    target : torch.Tensor
        One-hot encoded target of shape `(N, C, H, W)`.
    roi : torch.Tensor
        ROI of shape `(N, 1, H, W)`
    weight : torch.Tensor, optional
        Per class weight
    ignore_index : int, optional
        Specifies a target value that is ignored and does not contribute to the input gradient.
    eps : float
        Regularizer in the division

    Returns
    -------
    torch.Tensor
        Output as a torch.Tensor float
    """
    if weight is not None:
        raise NotImplementedError("Weight not yet implemented for dice loss.")

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")
    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"Input and target shapes must be the same. Got: {input.shape} and {target.shape}")
    if not input.device == target.device:
        raise ValueError(f"Input and target must be in the same device. Got: {input.device} and {target.device}")

    if ignore_index is not None:
        mask = target != ignore_index
        input = mask * input
        target = mask * target

    # Apply the ROI if it is there
    if roi is not None:
        # Correct the ROI to remove any unannotated regions (see github issue #11)
        background = (target[:, 0, :, :] == 1).int()
        corrected_roi = (roi.squeeze(1) - background).unsqueeze(1)
        # Clip the values below 0 after the subtraction.
        corrected_roi[corrected_roi < 0] = 0
        input = corrected_roi * input
        target = corrected_roi * target

    # Softmax still needs to be taken (logits are outputted by the network)
    input_soft = F.softmax(input, dim=1)
    # FIXME: This is not correct, when we input 0s for both target and input, we get 1.0.

    # Compute the dice score
    intersection = torch.sum(input_soft * target, dim=(2, 3))
    cardinality = torch.sum(input_soft**2, dim=(2, 3)) + torch.sum(target**2, dim=(2, 3))

    # dice_score has shape (B, C)
    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice_score.mean(axis=1)  # type: ignore
