"""Utilities to work with regions-of-interest. Has utilities to compute balanced ROIs from masks"""
from __future__ import annotations

import numpy as np
from dlup.annotations import WsiAnnotations
from dlup.tiling import TilingMode, tiles_grid_coordinates

from ahcore.utils.io import logger
from ahcore.utils.types import Rois


def compute_rois(
    mask: WsiAnnotations,
    tile_size: tuple[int, int],
    tile_overlap: tuple[int, int],
    centered: bool = True,
) -> Rois:
    """
    Compute the ROIs from a `WsiAnnotations` object. The ROIs are computed by:
      1) The bounding box of the whole `mask` object is computed (by `dlup`).
      2) The whole region up to the bounding box is computed.
      3) For each of the regions in the result, the bounding box is determined.

    If `centered` is True, the bounding box is centered in the region. This is done as follows:
      1) The effective region size is computed, depending on the tiling mode. This is the size of the region which
      is tiled by the given tile size and overlap. Tiles always overflow, so there are tiles on the border which are
      partially covered.
      2) The ROIs obtained in the first step are replaced by the effective region size, and centered around the original
      ROI.

    Parameters
    ----------
    mask : WsiAnnotations
    tile_size : tuple[int, int]
    tile_overlap : tuple[int, int]
    centered : bool

    Returns
    -------
    list[tuple[tuple[int, int], tuple[int, int]]]
        List of ROIs (coordinates, size).

    """
    bbox_coords, bbox_size = mask.bounding_box
    logger.debug("Annotations bounding box: %s, %s", bbox_coords, bbox_size)
    total_roi = mask.read_region((0, 0), 1.0, (bbox_coords[0] + bbox_size[0], bbox_coords[1] + bbox_size[1]))

    _rois = np.asarray([_.bounds for _ in total_roi])
    _rois[:, 2:] = _rois[:, 2:] - _rois[:, :2]
    rois = [((roi[0], roi[1]), (roi[2], roi[3])) for roi in _rois]
    logger.debug("Regions of interest: %s", rois)

    if centered:
        centered_rois = _get_centered_rois(rois, tile_size, tile_overlap)
        logger.debug("Centered ROIs: %s", centered_rois)
        return centered_rois

    return rois


def _get_centered_rois(roi_boxes: Rois, tile_size: tuple[int, int], tile_overlap: tuple[int, int]) -> Rois:
    """
    Based on the ROI and the effective region size compute a grid aligned at the center of the region

    Parameters
    ----------
    roi_boxes :
        The effective roi boxes
    tile_size : tuple[int, int]
        The size of the tiles which will be used to compute the balanced ROIs
    tile_overlap : tuple[int, int]
        The tile overlap of the tiles in the dataset.

    Returns
    -------
    list[tuple[tuple[int, int], tuple[int, int]]]
        List of ROIs (coordinates, size).

    """
    logger.debug("Computing balanced ROIs from ROI boxes %s", roi_boxes)
    region_sizes = [_compute_effective_size(roi_size, tile_size, tile_overlap) for _, roi_size in roi_boxes]

    output_rois: Rois = []
    for roi, region_size in zip(roi_boxes, region_sizes):
        offset, size = roi
        _region_size = np.asarray(region_size)
        _offset = np.asarray(offset)
        _size = np.asarray(size)

        _new_offset = _offset - (_region_size - _size) / 2
        _coordinates = int(_new_offset[0]), int(_new_offset[1])
        output_rois.append((_coordinates, region_size))
    return output_rois


def _compute_effective_size(
    size: tuple[int, int],
    tile_size: tuple[int, int],
    tile_overlap: tuple[int, int],
    mode: TilingMode = TilingMode.overflow,
) -> tuple[int, int]:
    """
    Compute the effective size of a tiled region, depending on the tiling mode and given the size. The effective size
    basically is the size of the region which is tiled by the given tile size and overlap. If tiles overflow, there are
    tiles on the border which are partially covered. If tiles are centered, the effective size of the region will be
    smaller than the original size.

    Parameters
    ----------
    size : tuple[int, int]
    tile_size : tuple[int, int]
    tile_overlap : tuple[int, int]
    mode : TilingMode

    Returns
    -------
    tuple[int, int]
        The effective size of the grid
    """
    coordinates_x, coordinates_y = tiles_grid_coordinates(
        size=size, tile_size=tile_size, tile_overlap=tile_overlap, mode=mode
    )
    effective_size = (
        coordinates_x.max() + tile_size[0],
        coordinates_y.max() + tile_size[1],
    )
    return effective_size
