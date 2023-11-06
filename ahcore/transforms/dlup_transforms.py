# Copyright (c) dlup contributors
# pylint: disable=unsubscriptable-object
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, cast

import cv2
import dlup.annotations
import numpy as np
import numpy.typing as npt
from dlup._exceptions import AnnotationError
from dlup.annotations import AnnotationType
from dlup.data.dataset import BoundingBoxType, PointType, TileSample, TileSampleWithAnnotationData

_AnnotationsTypes = dlup.annotations.Point | dlup.annotations.Polygon


def convert_point_annotations(
    annotations: Iterable[_AnnotationsTypes],
    region_size: tuple[int, int],
    index_map: dict[str, int],
    roi_name: str | None = None,
    default_value: int = 0,
    radius: int = 15,
) -> tuple[
    dict[str, list[PointType]],
    dict[str, list[BoundingBoxType]],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_] | None,
]:
    """
    Convert the polygon and point annotations as output of a dlup dataset class, where:
    - In case of points the output is dictionary mapping the annotation name to a list of locations. Point annotations
    will also subsequently be converted into polygons with size of `radius`.
    - In case of polygons these are converted into a mask according to `index_map`.

    *BE AWARE*: the polygon annotations are processed sequentially and later annotations can overwrite earlier ones.
    This is for instance useful when you would annotate "tumor associated stroma" on top of "stroma".
    The dlup Annotation classes return the polygons with area from large to small.

    When the polygon has holes, the previous written annotation is used to fill the holes.

    TODO
    ----
    - Convert segmentation index map to an Enum
    - Do we need to return PIL images here? If we load a tif mask the mask will be returned as a PIL image, so
      for consistency it might be relevant to do the same here.

    Parameters
    ----------
    annotations
    region_size : tuple[int, int]
    index_map : dict[str, int]
        Map mapping annotation name to index number in the output.
    roi_name : str
        Name of the region-of-interest key.
    default_value : int
        The mask will be initialized with this value.
    radius : int
        Radius in pixels which converts point annotations to polygons

    Returns
    -------
    dict, np.ndarray, np.ndarray or None
        Dictionary of points, mask and roi_mask.

    """
    mask = np.empty(region_size, dtype=np.int32)
    mask[:] = default_value
    points: dict[str, list[PointType]] = defaultdict(list)
    boxes: dict[str, list[BoundingBoxType]] = defaultdict(list)

    roi_mask = np.zeros(region_size, dtype=np.int32)
    has_roi = False
    for curr_annotation in annotations:
        holes_mask = None
        if isinstance(curr_annotation, dlup.annotations.Point):
            points[curr_annotation.label] += tuple(curr_annotation.coords)
            a_cls = dlup.annotations.AnnotationClass(label=curr_annotation.label, a_cls=AnnotationType.POLYGON)
            curr_annotation = dlup.annotations.Polygon(curr_annotation.buffer(radius), a_cls=a_cls)

        if isinstance(curr_annotation, dlup.annotations.Polygon) and curr_annotation.type == AnnotationType.BOX:
            min_x, min_y, max_x, max_y = curr_annotation.bounds
            boxes[curr_annotation.label].append(((int(min_x), int(min_y)), (int(max_x - min_x), int(max_y - min_y))))

        if roi_name and curr_annotation.label == roi_name:
            cv2.fillPoly(
                roi_mask,
                [np.asarray(curr_annotation.exterior.coords).round().astype(np.int32)],
                1,
            )
            has_roi = True
            continue

        if not (curr_annotation.label in index_map):
            continue

        original_values = None
        interiors = [np.asarray(pi.coords).round().astype(np.int32) for pi in curr_annotation.interiors]
        if interiors is not []:
            original_values = mask.copy()
            holes_mask = np.zeros(region_size, dtype=np.int32)
            # Get a mask where the holes are
            cv2.fillPoly(holes_mask, interiors, 1)

        cv2.fillPoly(
            mask,
            [np.asarray(curr_annotation.exterior.coords).round().astype(np.int32)],
            index_map[curr_annotation.label],
        )
        if interiors is not []:
            # TODO: This is a bit hacky to ignore mypy here, but I don't know how to fix it.
            mask = np.where(holes_mask == 1, original_values, mask)  # type: ignore

    # This is a hard to find bug, so better give an explicit error.
    if not has_roi and roi_name is not None:
        raise AnnotationError(f"ROI mask {roi_name} not found, please add a ROI mask to the annotations.")

    return dict(points), dict(boxes), mask, roi_mask if roi_name else None


class ConvertPointAnnotationsToMask:
    """Transform which converts polygons to masks. Will overwrite the annotations key"""

    def __init__(
        self,
        *,
        roi_name: str | None,
        index_map: dict[str, int],
        radius: int = 15,
        default_value: int = 0,
    ):
        """
        Parameters
        ----------
        roi_name : str, optional
            Name of the ROI key.
        index_map : dict
            Dictionary mapping the label to the integer in the output.
        radius : int
            Radius in pixels for converting point annotations to polygons
        default_value : int
            The mask will be initialized with this value.
        """
        self._roi_name = roi_name
        self._index_map = index_map
        self._radius = radius
        self._default_value = default_value

    def __call__(self, sample: TileSample) -> TileSampleWithAnnotationData:
        if not sample["annotations"]:
            raise ValueError("No annotations found to convert to mask.")

        _annotations = sample["annotations"]
        points, boxes, mask, roi = convert_point_annotations(
            _annotations,
            sample["image"].size[::-1],
            roi_name=self._roi_name,
            index_map=self._index_map,
            default_value=self._default_value,
            radius=self._radius,
        )

        output: TileSampleWithAnnotationData = cast(TileSampleWithAnnotationData, sample)
        output["annotation_data"] = {
            "points": points,
            "boxes": boxes,
            "mask": mask,
            "roi": roi,
        }

        return output
