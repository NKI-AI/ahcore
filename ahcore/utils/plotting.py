# encoding: utf-8
"""Plotting utilities for logging."""
# from __future__ import annotations

# from typing import Iterable

# import numpy as np
# import PIL
# import PIL.Image
# import PIL.ImageColor
# import PIL.ImageDraw
# import torch
# from dlup.annotations import AnnotationClass, AnnotationType, Point, Polygon
# from torchvision.transforms import functional as TF  # noqa
# from torchvision.utils import make_grid


# def plot_2d(
#     image: PIL.Image.Image,
#     mask: np.ndarray | None = None,
#     mask_colors=None,
#     alpha=70,
#     geometries=None,
#     geometries_color_map=None,
#     geometries_width: int = 2,
# ) -> PIL.Image.Image:
#     """
#     Plotting utility to overlay masks and geometries (Points, Polygons) on top of the image.

#     Parameters
#     ----------
#     image : PIL.Image
#     mask : np.ndarray
#         Integer array
#     mask_colors : dict
#         A dictionary mapping the integer value of the mask to a PIL color value.
#     alpha : int
#         Value between 0-100 defining the transparency of the overlays
#     geometries : list
#         A list of `Point`'s or `Polygon`'s
#     geometries_color_map : dict
#         Dictionary mapping label names to the PIL color value
#     geometries_width : int
#         Thickness of the contours and other lines

#     Returns
#     -------
#     PIL.Image
#     """
#     if image is None and mask is None:
#         raise RuntimeError("Not both `image` and `mask` can be empty")
#     if image is None:
#         image = PIL.Image.new("RGBA", mask.shape[-2:])

#     image = image.convert("RGBA")

#     def _get_geometries_color(label):
#         if geometries_color_map is None:
#             return None
#         return geometries_color_map.get(label, None)

#     if mask_colors is None:
#         pass

#     if mask is not None:
#         # Get unique values
#         unique_vals = sorted(list(np.unique(mask.astype(int))))
#         for idx in unique_vals:
#             if idx == 0:
#                 continue
#             color = PIL.ImageColor.getcolor(mask_colors[idx], "RGBA")
#             curr_mask = PIL.Image.fromarray(((mask == idx)[..., np.newaxis] * color).astype(np.uint8), mode="RGBA")
#             alpha_channel = PIL.Image.fromarray(((mask == idx) * int(alpha * 255 / 100)).astype(np.uint8), mode="L")
#             curr_mask.putalpha(alpha_channel)
#             image = PIL.Image.alpha_composite(image.copy(), curr_mask.copy()).copy()

#     if geometries is not None:
#         draw = PIL.ImageDraw.Draw(image)
#         for data in geometries:
#             if isinstance(data, Point):
#                 r = 10
#                 x, y = data.coords[0]
#                 _points = [(x - r, y - r), (x + r, y + r)]
#                 draw.ellipse(
#                     _points, outline=_get_geometries_color(data.annotation_class.label), width=geometries_width
#                 )

#             elif isinstance(data, Polygon):
#                 coordinates = data.exterior.coords
#                 draw.polygon(
#                     coordinates,
#                     fill=None,
#                     outline=_get_geometries_color(data.annotation_class.label),
#                     width=geometries_width,
#                 )

#             else:
#                 raise RuntimeError(f"Type {type(data)} not implemented.")

#     return image.convert("RGB")


# def binary_mask_to_polygon(mask: np.ndarray, label: str | None = None) -> list[Polygon]:
#     geometries = []
#     contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     # TODO: This can be done easier
#     contours = map(np.squeeze, contours)
#     for contour in contours:
#         if len(contour) < 3:
#             continue
#         geometries.append(Polygon(contour, label=label))
#     return geometries


# def one_hot_mask_to_polygon(mask: np.ndarray, index_map: dict[str, int] | None, threshold: float = 0.5):
#     _index_map_inv = {v: k for k, v in index_map.items()} if index_map else {}  # Index to label
#     geometries = []
#     for idx, curr_mask in enumerate(mask):
#         if idx == 0:  # Skip background
#             continue
#         label = _index_map_inv.get(idx, None)
#         geometries += binary_mask_to_polygon((curr_mask >= threshold).astype(int), label=label)
#     return geometries


# def plot_batch(
#     image_batch: torch.Tensor,
#     mask_batch: torch.Tensor | None,
#     point_predictions: torch.Tensor | None,
#     index_map: dict[str, int] | None = None,
#     colors: dict[str, str] | None = None,
#     mask_as_polygon: bool = False,
# ) -> torch.Tensor:
#     if index_map is None:
#         index_map = {}
#     if colors is None:
#         colors = {}

#     rev_index_map = {v: k for k, v in index_map.items()}

#     #  Convert colors to mask index, ignore index 0 (background)
#     # mask_colors = {v: colors[k] for k, v in index_map.items() if v != 0}
#     mask_colors = {0: "blue", 1: "green", 2: "red"}

#     _mask_batch: Iterable
#     if isinstance(mask_batch, torch.Tensor):
#         _mask_batch = mask_batch
#     else:
#         _mask_batch = [None] * image_batch.shape[0]

#     _point_predictions: Iterable
#     if isinstance(point_predictions, torch.Tensor):
#         _point_predictions = point_predictions
#     else:
#         _point_predictions = [None] * image_batch.shape[0]

#     images = []
#     for i, (image, mask, points) in enumerate(zip(image_batch, _mask_batch, _point_predictions)):
#         pil_image = TF.to_pil_image(image.cpu())

#         alpha = 60
#         _mask = None
#         geometries = None
#         if mask is not None:
#             mask = mask.detach().cpu().numpy()
#             _mask = mask.copy()
#             if _mask.ndim == 3:  # Not yet one hot encoded
#                 _mask = _mask.argmax(axis=0)
#             # geometries = None
#             if mask_as_polygon:
#                 raise ValueError("DON'T USE THIS")
#             #     # geometries = one_hot_mask_to_polygon(mask, index_map, threshold=0.5)
#             #     alpha = 70
#             # else:
#             #     alpha = 70

#             # if multipoints_image is not None:
#             #     if geometries is None:
#             #         geometries = []
#             #     for label, mps in multipoints_image.items():
#             #         geometries.extend([Point(p.xy, label) for p in mps[i].geoms])
#         if points is not None:
#             _points = points.cpu().numpy()
#             if geometries is None:
#                 geometries = []
#             for point in _points:
#                 # Break when padding points are encountered
#                 if point[0] == torch.inf:
#                     continue
#                 label = rev_index_map[int(point[2])] if rev_index_map is not None else "point_pred"
#                 geometries.append(Point(point[:2], a_cls=AnnotationClass(label, AnnotationType.POINT)))

#         image_with_overlay = plot_2d(
#             pil_image,
#             mask=_mask,
#             alpha=alpha,
#             mask_colors=mask_colors,
#             geometries=geometries,
#             geometries_color_map=colors,
#         )
#         images.append(TF.pil_to_tensor(image_with_overlay))

#     # Convert the image tensor list into one stacked along the batch again

#     image_grid = make_grid(torch.stack(images))
#     return image_grid


# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from typing import Optional
# from torch import Tensor


# from PIL import Image
# import io

# def plot_batch(
#     batch: dict[str, Tensor],
#     prediction: Tensor,
#     point_predictions: Optional[Tensor] = None,
#     custom_cmap_list: Optional[list[str]] = None,
# ):
#     batch_image = batch["image"].permute(0, 2, 3, 1).detach().cpu().numpy()
#     batch_target = batch["target"].permute(0, 2, 3, 1).detach().cpu().numpy()

#     prediction_np = prediction.permute(0, 2, 3, 1).detach().cpu().numpy()

#     if point_predictions is not None:
#         batch_points = batch["points"].detach().cpu().numpy()
#         batch_labels = batch["points_labels"].detach().cpu().numpy()

#     if custom_cmap_list is not None:
#         custom_cmap = ListedColormap(custom_cmap_list)
#     else:
#         custom_cmap = None

#     fig, axs = plt.subplots(len(batch_image), 3, figsize=(len(batch_image) * 1.5, 6 * len(batch_image)))
#     for i in range(len(batch_image)):
#         axs[i, 0].imshow(batch_image[i])
#         axs[i, 1].imshow(batch_target[i])
#         axs[i, 2].imshow(prediction_np[i])
#         if point_predictions is not None:
#             axs[i, 2].scatter(
#                 x=batch_points[i, :, 0], y=batch_points[i, :, 1], c=batch_labels[i], cmap=custom_cmap, s=15
#             )
#     plt.tight_layout()
#     image_buffer = io.BytesIO()
#     fig.savefig(image_buffer, format="png")
#     image_buffer.seek(0)
#     image = plt.imread(image_buffer)
#     return image
