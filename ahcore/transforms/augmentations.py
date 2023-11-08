"""
Augmentations factory
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union, cast

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
from kornia.augmentation import random_generator as rg
from kornia.constants import DataKey, Resample
from kornia.geometry import transform_bbox, transform_points
from omegaconf import ListConfig
from torch import nn

from ahcore.utils.data import DataDescription
from ahcore.utils.io import get_logger

logger = get_logger(__name__)


def validate_sample_annotations(
    sample: dict[str, torch.Tensor], ann_type: Literal["points", "boxes"] = "points"
) -> dict[str, torch.Tensor]:
    """Validate annotations after augmentations based on image size and roi (if it exists).

    Parameters
    ----------
    sample : dict[str, torch.Tensor]
        Sample after augmentations. Image size is determined by "image" tensor. Optionally, validity of annotations is
        checked if "roi" tensor is found.
    ann_type : Literal[&quot;points&quot;, &quot;boxes&quot;], optional
        Annotation type used to validate, by default "points"

    Returns
    -------
    dict[str, torch.Tensor]
        Sample with invalid annotation and labels set to torch.nan and -1 respectively.

    Raises
    ------
    NotImplementedError
        Validation of ann_type `boxes` has not been implemented yet.
    """
    if ann_type == "boxes":
        raise NotImplementedError

    # Validate if annotations are within bounds
    _size = sample["image"].shape[-2:]
    _annotations = sample[ann_type]
    valid_idx = ((_annotations >= 0) & (_annotations < _size[0])).all(dim=2)
    _annotations[~valid_idx] = torch.nan

    # Validate annotations are within ROI
    _roi = sample.get("roi")
    if _roi is not None:
        _roi = _roi.squeeze()
        # TODO: Only taking valid_idx and not padded values
        _first_idx = torch.arange(_annotations.shape[0]).repeat_interleave(_annotations.shape[1])
        _annotations_idx = _annotations.round().long()
        roi_valid = _roi[_first_idx, _annotations_idx[:, :, 0].flatten(), _annotations_idx[:, :, 1].flatten()]
        valid_idx = valid_idx & roi_valid.reshape(-1, _annotations.shape[1]).bool()
        _annotations[~valid_idx] = torch.nan

    _labels = sample[f"{ann_type}_labels"]
    _labels[~valid_idx] = -1
    return sample


class MeanStdNormalizer(nn.Module):
    """
    Normalizes the mean and standard deviation of the input image. Assumes the original range is `[0, 255]`.
    """

    def __init__(
        self,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
    ):
        """
        Parameters
        ----------
        mean : tuple[float, float, float], optional
        std : tuple[float, float, float], optional
        """
        super().__init__()
        if mean is None:
            self._mean = nn.Parameter(torch.Tensor([0.0] * 3), requires_grad=False)
        else:
            self._mean = nn.Parameter(torch.Tensor(mean), requires_grad=False)

        if std is None:
            self._std = nn.Parameter(torch.Tensor([1.0] * 3), requires_grad=False)
        else:
            self._std = nn.Parameter(torch.Tensor(std), requires_grad=False)

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> list[torch.Tensor] | torch.Tensor:
        output = []
        data_keys = kwargs["data_keys"]
        for sample, data_key in zip(args, data_keys):
            if data_key in [DataKey.INPUT, 0, "INPUT"]:
                sample = sample / 255.0
                sample = (sample - self._mean[..., None, None].to(sample.device)) / self._std[..., None, None].to(
                    sample.device
                )
            output.append(sample)

        if len(output) == 1:
            return output[0]

        return output


class HEDColorAugmentation(K.IntensityAugmentationBase2D):
    """
    A torch implementation of the color stain augmentation algorithm on the
    deconvolved Hemaetoxylin-Eosin-DAB (HED) channels of an image as described
    by Tellez et al. (2018) in Appendix A & B here: https://arxiv.org/pdf/1808.05896.pdf.
    """

    # Normalized OD matrix from Ruifrok et al. (2001)
    HED_REFERENCE = torch.Tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])

    def __init__(
        self,
        scale_sigma: float | list[float] | ListConfig,
        bias_sigma: float | list[float] | ListConfig,
        epsilon: float = 1e-6,
        clamp_output_range: Optional[tuple[float, float]] = (0, 1),
        p: float = 0.5,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Apply a color stain augmentation in the Hemaetoxylin-Eosin-DAB (HED) color space based on [1].
        The fixed normalized OD matrix values are based on [2].

        Parameters
        ----------
        scale_sigma: float, ListConfig or list of floats
            For each channel in the HED space a random scaling factor is drawn from alpha_i ~ U(1-sigma_i,1+sigma_i).
        bias_sigma: float, ListConfig or list of floats
            For each channel in the HED space a random bias is added drawn from beta_i ~ U(-sigma_i,sigma_i).
        epsilon: float
            Small positive bias to avoid numerical errors
        clamp_output_range: tuple of floats or None
            Clamp output in range after augmenting the input. Conventionally a range of [0, 1] is used, but this will
            discard some color information. `scale_sigma` and `bias_sigma` should be chosen carefully to prevent this.

        NOTE: To reproduce augmentations as shown in [3], a larger positive bias of `epsilon=2` in combination with
        `clamp_output_range=(0, 1)` should be used. Then, use `scale_sigma=bias_sigma=0.05` for `HED-light` and
        `scale_sigma=bias_sigma=0.2` for `HED-strong`.
        When using the default `epsilon=1e-6` and `clamp_output_range=(0, 1)`, this is not equal but comparable to
        `scale_sigma=bias_sigma=0.15` for `HED-light` and `scale_sigma=bias_sigma=0.8` for `HED-strong`?


        References
        ----------
        [1] Tellez, David, et al. "Whole-slide mitosis detection in H&E breast histology using PHH3 as
            a reference to train distilled stain-invariant convolutional networks."
            IEEE transactions on medical imaging 37.9 (2018): 2126-2136.
        [2] Ruifrok AC, Johnston DA. Quantification of histochemical staining by color deconvolution.
            Anal Quant Cytol Histol. 2001 Aug;23(4):291-9. PMID: 11531144.
        [3] Tellez, David, et al. "Quantifying the effects of data augmentation and stain color normalization
            in convolutional neural networks for computational pathology."
            Medical image analysis 58 (2019): 101544.
        """
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)
        if (isinstance(scale_sigma, (list, ListConfig)) and len(scale_sigma) != 3) or (
            isinstance(bias_sigma, (list, ListConfig)) and len(bias_sigma) != 3
        ):
            raise ValueError(
                f"scale_sigma and bias_sigma should have either 1 or 3 values, "
                f"got {scale_sigma} and {bias_sigma} instead."
            )

        if isinstance(scale_sigma, float):
            scale_sigma = [scale_sigma for _ in range(3)]

        if isinstance(bias_sigma, float):
            bias_sigma = [bias_sigma for _ in range(3)]

        _scale_sigma = torch.tensor(scale_sigma)
        _bias_sigma = torch.tensor(bias_sigma)

        scale_factor = torch.stack([1 - _scale_sigma, 1 + _scale_sigma])
        bias_factor = torch.stack([-_bias_sigma, _bias_sigma])

        self._param_generator = rg.PlainUniformGenerator(
            (scale_factor, "scale", None, None), (bias_factor, "bias", None, None)
        )
        self.flags = {
            "epsilon": torch.tensor([epsilon]),
            "M": self.HED_REFERENCE,
            "M_inv": torch.linalg.inv(self.HED_REFERENCE),
            "clamp_output_range": clamp_output_range,
        }

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply HED color augmentation on an input tensor.
        """
        assert flags, "Flags should be provided"
        assert params, "Params should be provided"

        epsilon = flags["epsilon"].to(input)
        reference_matrix = flags["M"].to(input)
        reference_matrix_inv = flags["M_inv"].to(input)
        alpha = params["scale"][:, None, None, :].to(input)
        beta = params["bias"][:, None, None, :].to(input)

        rgb_tensor = input.permute(0, 2, 3, 1)
        optical_density = -torch.log(rgb_tensor + epsilon)
        # Mypy doesn't understand this is a tensor.
        hed_tensor = cast(torch.Tensor, optical_density @ reference_matrix_inv)

        augmented_hed_tensor = alpha * hed_tensor + beta
        # Same problem that mypy doesn't understand
        augmented_rgb_tensor = cast(torch.Tensor, torch.exp(-augmented_hed_tensor @ reference_matrix) - epsilon)
        augmented_sample = augmented_rgb_tensor.permute(0, 3, 1, 2)

        if flags["clamp_output_range"] is not None:
            augmented_sample = torch.clamp(augmented_sample, *flags["clamp_output_range"])
        return augmented_sample


class CenterCrop(nn.Module):
    """Perform a center crop of the image and target"""

    def __init__(self, size: int | tuple[int, int], **kwargs: Any) -> None:
        super().__init__()
        _size = size
        if isinstance(size, int):
            _size = (size, size)

        if isinstance(size, ListConfig) or isinstance(size, list):
            _size = tuple(size)

        self._cropper = K.CenterCrop(size=_size, align_corners=True, p=1.0, keepdim=False, cropping_mode="slice")

    def forward(
        self,
        *sample: torch.Tensor,
        data_keys: Optional[list[str | int | DataKey]] = None,
        **kwargs: Any,
    ) -> Union[list[torch.Tensor], torch.Tensor]:
        if data_keys is None:
            output = [cast(torch.Tensor, self._cropper(item)) for item in sample]
        else:
            output = []
            for item, data_key in zip(sample, data_keys):
                if data_key in [DataKey.INPUT, 0, "INPUT", DataKey.MASK, 1, "MASK"]:
                    output.append(cast(torch.Tensor, self._cropper(item)))
                else:
                    trans_mat = self._cropper.get_transformation_matrix(item)
                    if data_key in [DataKey.KEYPOINTS, 5, "KEYPOINTS"]:
                        output.append(transform_points(trans_mat, item))
                    else:
                        output.append(transform_bbox(trans_mat, item, mode="xywh"))

        if len(output) == 1:
            return output[0]
        return output


def _parse_random_apply(random_apply: int | bool | tuple[int, int] | ListConfig) -> int | bool | tuple[int, int]:
    if isinstance(random_apply, (int, bool)):
        return random_apply

    if isinstance(random_apply, ListConfig) or isinstance(random_apply, list):
        return cast(tuple[int, int], tuple(random_apply))

    return random_apply


def _parse_random_apply_weights(
    random_apply_weights: list[float] | ListConfig | None,
) -> list[float] | None:
    if isinstance(random_apply_weights, ListConfig) or isinstance(random_apply_weights, list):
        return cast(list[float], list(random_apply_weights))

    return random_apply_weights


class AugmentationFactory(nn.Module):
    """Factory for the augmentation. There are three classes of augmentations:
    - `initial_transforms`: Transforms which are the first to always be applied to the sample
    - `intensity_augmentations`: Transforms which only affect the intensity and not the geometry. Only applied to the
    image.
    - `geometric_augmentations`: Transforms which affect the geometry. They are applied to both the image, ROI and mask.
    """

    DATA_KEYS = {
        "image": DataKey.INPUT,
        "target": DataKey.MASK,
        "roi": DataKey.MASK,
        "points": DataKey.KEYPOINTS,
        "boxes": DataKey.BBOX_XYWH,
    }

    def __init__(
        self,
        data_description: DataDescription,
        data_module: pl.LightningDataModule,
        initial_transforms: list[nn.Module] | None = None,
        random_apply_intensity: int | bool | ListConfig = False,
        random_apply_weights_intensity: list[float] | None = None,
        intensity_augmentations: list[K.IntensityAugmentationBase2D] | None = None,
        random_apply_geometric: int | bool | ListConfig = False,
        random_apply_weights_geometric: list[float] | ListConfig | None = None,
        geometric_augmentations: list[K.GeometricAugmentationBase2D] | None = None,
        final_transforms: list[nn.Module] | None = None,
    ) -> None:
        super().__init__()

        self._transformable_keys = ["image", "target"]
        self._annotation_keys: list[Literal["points", "boxes"]] = []
        if data_description.use_roi:
            self._transformable_keys.append("roi")
        if data_description.use_points:
            self._transformable_keys.append("points")
            self._annotation_keys.append("points")
        if data_description.use_boxes:
            self._transformable_keys.append("boxes")
            self._annotation_keys.append("boxes")

        if (
            any(ann_type in self._annotation_keys for ann_type in ["points", "boxes"])
            and "roi" not in self._transformable_keys
            and geometric_augmentations is not None
        ):
            logger.warn(
                "Using keypoints and boxes as annotations without ROIs might result in invalid annotations because of \
                geometric augmentations."
            )
        self._data_keys = [self.DATA_KEYS[key] for key in self._transformable_keys]

        # Initial transforms will be applied sequentially
        if initial_transforms:
            for transform in initial_transforms:
                logger.info("Using initial transform %s", transform)
        self._initial_transforms = nn.ModuleList(initial_transforms)

        # Intensity augmentations will be selected in random order
        if intensity_augmentations:
            for transform in intensity_augmentations:
                logger.info("Adding intensity augmentation %s", transform)

        self._intensity_augmentations = None
        if intensity_augmentations:
            self._intensity_augmentations = K.AugmentationSequential(
                *intensity_augmentations,
                data_keys=list(self.DATA_KEYS.values()),
                same_on_batch=False,
                random_apply=_parse_random_apply(random_apply_intensity),
                random_apply_weights=_parse_random_apply_weights(random_apply_weights_intensity),
            )

        # Geometric augmentations will be selected in random order.
        if geometric_augmentations:
            for transform in geometric_augmentations:
                logger.info("Adding geometric augmentation %s", transform)

        self._geometric_augmentations = None
        if geometric_augmentations:
            self._geometric_augmentations = K.AugmentationSequential(
                *geometric_augmentations,
                data_keys=list(self.DATA_KEYS.values()),
                same_on_batch=False,
                random_apply=_parse_random_apply(random_apply_geometric),
                random_apply_weights=_parse_random_apply_weights(random_apply_weights_geometric),
                extra_args={DataKey.MASK: dict(resample=Resample.NEAREST, align_corners=True)},
            )

        # Final transforms will be applied sequentially
        if final_transforms:
            for transform in final_transforms:
                logger.info("Using final transform %s", transform)
        self._final_transforms = nn.ModuleList(final_transforms)

    def forward(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output_data = [sample[key] for key in self._transformable_keys if key in sample]
        if self._initial_transforms:
            kwargs = {
                "data_keys": self._data_keys,
                "filenames": sample["path"],
            }
            for transform in self._initial_transforms:
                output_data = transform(*output_data, **kwargs)

        if isinstance(output_data, torch.Tensor):
            output_data = [output_data]

        if self._intensity_augmentations:
            output_data[0] = self._intensity_augmentations(*output_data[:1], data_keys=[DataKey.INPUT])

        if self._geometric_augmentations:
            output_data = self._geometric_augmentations(*output_data, data_keys=self._data_keys)

        if isinstance(output_data, torch.Tensor):
            output_data = [output_data]

        if self._final_transforms:
            for transform in self._final_transforms:
                output_data = transform(*output_data, data_keys=self._data_keys)

        if isinstance(output_data, torch.Tensor):
            output_data = [output_data]

        # Add the output data back into the sample
        for key, curr_output in zip(self._transformable_keys, output_data):
            sample[key] = curr_output

        # Verify annotation output (using ROI if it exists in sample)
        if self._annotation_keys:
            for ann_type in self._annotation_keys:
                sample = validate_sample_annotations(sample, ann_type=ann_type)
        return sample
