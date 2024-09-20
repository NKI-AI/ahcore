from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Tuple

from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.background import compute_masked_indices
from dlup.tiling import Grid
from pymilvus.client.abstract import Hit, SearchResult
from shapely.geometry import MultiPolygon

from ahcore.utils.data import DataDescription
from ahcore.utils.vector_database.plot_utils import plot_multipolygons
from ahcore.utils.vector_database.utils import (
    create_wsi_annotation_from_hits,
    create_wsi_annotation_from_regions,
    extract_results_per_wsi,
    generate_paths,
)

log = getLogger(__name__)


class MetricType(Enum):
    IOU = "iou"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"

    @classmethod
    def from_value(cls, value: str) -> "MetricType":
        """Convert a string to a MetricType enum if valid.

        Args:
            value (str): The string representation of the metric.

        Returns:
            MetricType: The corresponding MetricType enum.

        Raises:
            ValueError: If the metric string is invalid.
        """
        try:
            return MetricType(value.lower())
        except ValueError:
            raise ValueError(f"Invalid metric type: '{value}'. Valid metrics are: {[e.value for e in cls]}")


class Metrics:
    """
    A class to compute specified metrics for Whole Slide Images (WSIs) based on search results.

    Attributes:
        data_description (DataDescription): The data description containing directories and ROI information.
        metrics (List[MetricType]): A list of metrics to compute.
    """

    def __init__(
        self,
        data_dir: str,
        annotations_dir: str,
        annotation_label: str,
        metrics: List[str],
        tile_size: int = 224,
        annotation_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the Metrics class.

        Args:
            data_description (DataDescription): The data description object.
            metrics (List[str]): List of metrics to compute as strings.
        """
        self.metrics = [MetricType.from_value(m) for m in metrics]
        self.metric_functions = {
            MetricType.IOU: self.compute_iou,
            MetricType.PRECISION: self.compute_precision,
            MetricType.RECALL: self.compute_recall,
            MetricType.F1: self.compute_f1,
        }
        self.tile_size = tile_size  # Assuming square tiles
        self.annotation_threshold = annotation_threshold

        self.data_dir = data_dir
        self.annotations_dir = annotations_dir
        self.annotation_label = annotation_label

        self._max_distance = 15000  # Maximum distance used to avoid other slices in the same WSI
        self._aggregated_values: Dict[str, float] = {
            "total_intersection_area": 0.0,
            "total_union_area": 0.0,
            "total_hit_area": 0.0,
            "total_annotation_area": 0.0,
        }

    def compute_metrics(
        self, search_results: SearchResult, plot_results: bool = False
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Compute the specified metrics per WSI.

        Args:
            search_results (SearchResult): The search results from which to compute metrics.
            plot_results (bool, optional): Whether to plot the results. Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary with filenames as keys and metric dictionaries as values.
        """
        self._reset_aggregated_values()
        hits_per_wsi: Dict[str, List[Hit]] = extract_results_per_wsi(search_results)
        metrics_per_wsi: Dict[str, Dict[str, float]] = {}

        for filename, hits in hits_per_wsi.items():
            metrics, computed_values = self.compute_metrics_for_wsi(filename, hits, plot_results)
            metrics_per_wsi[filename] = metrics

            # Update the aggregated values
            self._update_aggregated_values(computed_values)

        global_metrics = self.compute_global_metrics()

        return metrics_per_wsi, global_metrics

    def _reset_aggregated_values(self) -> None:
        """
        Reset the aggregated values to zero.
        """
        self._aggregated_values = {
            "total_intersection_area": 0.0,
            "total_union_area": 0.0,
            "total_hit_area": 0.0,
            "total_annotation_area": 0.0,
        }

    def _update_aggregated_values(self, computed_values: Dict[str, Any]) -> None:
        """
        Update the aggregated values with the computed values.

        Args:
            computed_values (Dict[str, Any]): The computed values.
        """
        self._aggregated_values["total_intersection_area"] += computed_values.get("intersection_area", 0.0)
        self._aggregated_values["total_union_area"] += computed_values.get("union_area", 0.0)
        self._aggregated_values["total_hit_area"] += computed_values.get("hit_area", 0.0)
        self._aggregated_values["total_annotation_area"] += computed_values.get("annotation_area", 0.0)

    def compute_metrics_for_wsi(
        self, filename: str, hits: List[Hit], plot_results: bool
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute the specified metrics for a single WSI.

        Args:
            filename (str): The filename of the WSI.
            hits (List[Hit]): The list of hits for this WSI.
            plot_results (bool): Whether to plot the results.

        Returns:
            Dict[str, float]: A dictionary of computed metrics for this WSI.
        """
        # Load image and annotations
        image, annotation = self.load_image_and_annotations(filename)

        # Create hit annotations
        hit_annotations = self.create_hit_annotations(hits, image, annotation)

        # Prepare multipolygons
        hit_multipolygon, annotation_multipolygon = self.prepare_multipolygons(hit_annotations, annotation, image)

        # Compute metrics
        metrics: Dict[str, float] = {}
        computed_values: Dict[str, Any] = {}
        for metric in self.metrics:
            metric_value = self.metric_functions[metric](hit_multipolygon, annotation_multipolygon, computed_values)
            metrics[metric.value] = metric_value

        # Optionally plot results
        if plot_results:
            self.plot_results(image, annotation, hit_annotations, filename, hit_multipolygon, annotation_multipolygon)

        # Ensure the required areas are in computed_values for the aggregated values
        if "intersection_area" not in computed_values:
            computed_values["intersection_area"] = hit_multipolygon.intersection(annotation_multipolygon).area
        if "union_area" not in computed_values:
            computed_values["union_area"] = hit_multipolygon.union(annotation_multipolygon).area
        if "hit_area" not in computed_values:
            computed_values["hit_area"] = hit_multipolygon.area
        if "annotation_area" not in computed_values:
            computed_values["annotation_area"] = annotation_multipolygon.area

        return metrics, computed_values

    def compute_global_metrics(self) -> Dict[str, float]:
        """
        Compute global metrics over all WSIs.

        Returns:
            Dict[str, float]: Dictionary of global metrics.
        """
        aggregated_values = self._aggregated_values
        global_metrics = {}
        if MetricType.IOU in self.metrics:
            total_intersection_area = aggregated_values["total_intersection_area"]
            total_union_area = aggregated_values["total_union_area"]
            global_iou = total_intersection_area / total_union_area if total_union_area > 0 else 0.0
            global_metrics["iou"] = global_iou

        if MetricType.PRECISION in self.metrics:
            total_intersection_area = aggregated_values["total_intersection_area"]
            total_hit_area = aggregated_values["total_hit_area"]
            global_precision = total_intersection_area / total_hit_area if total_hit_area > 0 else 0.0
            global_metrics["precision"] = global_precision

        if MetricType.RECALL in self.metrics:
            total_intersection_area = aggregated_values["total_intersection_area"]
            total_annotation_area = aggregated_values["total_annotation_area"]
            global_recall = total_intersection_area / total_annotation_area if total_annotation_area > 0 else 0.0
            global_metrics["recall"] = global_recall

        if MetricType.F1 in self.metrics:
            global_precision = global_metrics.get("precision", 0.0)
            global_recall = global_metrics.get("recall", 0.0)
            if global_precision + global_recall > 0:
                global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall)
            else:
                global_f1 = 0.0
            global_metrics["f1"] = global_f1

        return global_metrics

    def load_image_and_annotations(self, filename: str) -> Tuple[SlideImage, WsiAnnotations]:
        """
        Load the image and annotations for a given filename.

        Args:
            filename (str): The filename of the WSI.

        Returns:
            Tuple[SlideImage, WsiAnnotations]: The loaded image and annotations.
        """
        image_filename, annotations_filename = generate_paths(
            filename,
            self.data_dir,
            self.annotations_dir,
        )
        image = SlideImage.from_file_path(image_filename, internal_handler="vips")
        annotation = WsiAnnotations.from_geojson(annotations_filename)
        annotation.filter(self.annotation_label)
        return image, annotation

    def create_hit_annotations(self, hits: List[Hit], image: SlideImage, annotation: WsiAnnotations) -> WsiAnnotations:
        """
        Create WSI annotations from the hits.

        Args:
            hits (List[Hit]): The list of hits.
            image (SlideImage): The slide image.
            annotation (WsiAnnotations): The WSI annotations.

        Returns:
            WsiAnnotations: The hit annotations.
        """
        hit_annotations = create_wsi_annotation_from_hits(
            hits, slide_image=image, annotation=annotation, max_distance=self._max_distance
        )
        return hit_annotations

    def prepare_multipolygons(
        self,
        hit_annotations: WsiAnnotations,
        annotation: WsiAnnotations,
        image: SlideImage,
    ) -> Tuple[MultiPolygon, MultiPolygon,]:
        """
        Prepare the multipolygons for hits and annotations, and compute the read bounding box.

        Args:
            hit_annotations (WsiAnnotations): The hit annotations.
            annotation (WsiAnnotations): The ground truth annotations.
            image (SlideImage): The slide image.

        Returns:
            Tuple[read_bbox, hit_multipolygon, annotation_multipolygon]
        """
        hit_bbox = hit_annotations.bounding_box
        annotation_bbox = self.adjust_bbox_to_multiple(annotation.bounding_box, self.tile_size)
        read_bbox = self.max_bbox(hit_bbox, annotation_bbox)

        # Create MultiPolygon for hits
        hit_multipolygon = MultiPolygon(
            hit_annotations.read_region(read_bbox[0], scaling=1.0, size=read_bbox[1])
        ).buffer(0)

        # check if annotation bbox is nonzero
        if annotation_bbox[0] == annotation_bbox[1]:
            annotation_multipolygon = MultiPolygon()

        else:
            # Grid for creating the annotation blocks
            grid = Grid.from_tiling(
                offset=annotation_bbox[0],
                size=annotation_bbox[1],
                tile_size=(self.tile_size, self.tile_size),
                tile_overlap=(0, 0),
                mode="overflow",
                order="C",
            )
            regions = [(x, y, self.tile_size, self.tile_size, 0.5) for x, y in grid]  # [x, y, width, height, mpp]
            masked_indices = compute_masked_indices(image, annotation, regions, threshold=self.annotation_threshold)
            boxed_annotation = create_wsi_annotation_from_regions(regions=regions, masked_indices=masked_indices)

            # Create MultiPolygon for annotations
            annotation_multipolygon = MultiPolygon(
                boxed_annotation.read_region(read_bbox[0], scaling=1.0, size=read_bbox[1])
            ).buffer(0)

        return hit_multipolygon, annotation_multipolygon

    def compute_iou(
        self,
        hit_multipolygon: MultiPolygon,
        annotation_multipolygon: MultiPolygon,
        computed_values: Dict[str, Any],
    ) -> float:
        """
        Compute the Intersection over Union (IoU) metric.

        Args:
            hit_multipolygon (MultiPolygon): The MultiPolygon of hit annotations.
            annotation_multipolygon (MultiPolygon): The MultiPolygon of ground truth annotations.
            computed_values (Dict[str, Any]): Dictionary to store intermediate computations.

        Returns:
            float: The IoU value.
        """
        if "intersection_area" not in computed_values:
            computed_values["intersection_area"] = hit_multipolygon.intersection(annotation_multipolygon).area
        if "union_area" not in computed_values:
            computed_values["union_area"] = hit_multipolygon.union(annotation_multipolygon).area
        intersection_area = computed_values["intersection_area"]
        union_area = computed_values["union_area"]
        iou = intersection_area / union_area if union_area > 0 else 0.0
        computed_values["iou"] = iou
        return iou

    def compute_precision(
        self,
        hit_multipolygon: MultiPolygon,
        annotation_multipolygon: MultiPolygon,
        computed_values: Dict[str, Any],
    ) -> float:
        """
        Compute the precision metric.

        Args:
            hit_multipolygon (MultiPolygon): The MultiPolygon of hit annotations.
            annotation_multipolygon (MultiPolygon): The MultiPolygon of ground truth annotations.
            computed_values (Dict[str, Any]): Dictionary to store intermediate computations.

        Returns:
            float: The precision value.
        """
        if "intersection_area" not in computed_values:
            computed_values["intersection_area"] = hit_multipolygon.intersection(annotation_multipolygon).area
        if "hit_area" not in computed_values:
            computed_values["hit_area"] = hit_multipolygon.area
        intersection_area = computed_values["intersection_area"]
        hit_area = computed_values["hit_area"]
        precision = intersection_area / hit_area if hit_area > 0 else 0.0
        computed_values["precision"] = precision
        return precision

    def compute_recall(
        self,
        hit_multipolygon: MultiPolygon,
        annotation_multipolygon: MultiPolygon,
        computed_values: Dict[str, Any],
    ) -> float:
        """
        Compute the recall metric.

        Args:
            hit_multipolygon (MultiPolygon): The MultiPolygon of hit annotations.
            annotation_multipolygon (MultiPolygon): The MultiPolygon of ground truth annotations.
            computed_values (Dict[str, Any]): Dictionary to store intermediate computations.

        Returns:
            float: The recall value.
        """
        if "intersection_area" not in computed_values:
            computed_values["intersection_area"] = hit_multipolygon.intersection(annotation_multipolygon).area
        if "annotation_area" not in computed_values:
            computed_values["annotation_area"] = annotation_multipolygon.area
        intersection_area = computed_values["intersection_area"]
        annotation_area = computed_values["annotation_area"]
        recall = intersection_area / annotation_area if annotation_area > 0 else 0.0
        computed_values["recall"] = recall
        return recall

    def compute_f1(
        self,
        hit_multipolygon: MultiPolygon,
        annotation_multipolygon: MultiPolygon,
        computed_values: Dict[str, Any],
    ) -> float:
        """
        Compute the F1 score.

        Args:
            hit_multipolygon (MultiPolygon): The MultiPolygon of hit annotations.
            annotation_multipolygon (MultiPolygon): The MultiPolygon of ground truth annotations.
            computed_values (Dict[str, Any]): Dictionary to store intermediate computations.

        Returns:
            float: The F1 score.
        """
        # Ensure precision and recall are computed
        if "precision" not in computed_values:
            precision = self.compute_precision(hit_multipolygon, annotation_multipolygon, computed_values)
            computed_values["precision"] = precision
        else:
            precision = computed_values["precision"]

        if "recall" not in computed_values:
            recall = self.compute_recall(hit_multipolygon, annotation_multipolygon, computed_values)
            computed_values["recall"] = recall
        else:
            recall = computed_values["recall"]

        # Compute F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return f1

    def plot_results(
        self,
        image: SlideImage,
        annotation: WsiAnnotations,
        hit_annotations: WsiAnnotations,
        filename: str,
        hit_multipolygon: MultiPolygon,
        annotation_multipolygon: MultiPolygon,
    ) -> None:
        """
        Plot the results.

        Args:
            image (SlideImage): The slide image.
            annotation (WsiAnnotations): The ground truth annotations.
            hit_annotations (WsiAnnotations): The hit annotations.
            filename (str): The filename of the WSI.
        """
        plot_multipolygons(
            hit_multipolygon,
            annotation_multipolygon,
        )
        # plot_wsi_and_annotation_overlay(
        #     image,
        #     annotation,
        #     hit_annotations,
        #     mpp=16,
        #     tile_size=(7, 7),
        #     filename_appendage=filename,
        # )

    @staticmethod
    def adjust_bbox_to_multiple(
        bbox: Tuple[Tuple[float, float], Tuple[float, float]], multiple: int
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Adjust a bounding box to be a multiple of a given number.

        Args:
            bbox (Tuple[Tuple[float, float], Tuple[float, float]]): The bounding box to adjust.
            multiple (int): The multiple to adjust to.

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: The adjusted bounding box.
        """

        def lower_multiple(value: float, multiple: int) -> float:
            return (value // multiple) * multiple

        def upper_multiple(value: float, multiple: int) -> float:
            return ((value + multiple - 1) // multiple) * multiple

        start_x, start_y = bbox[0]
        end_x, end_y = bbox[1]

        new_start_x = lower_multiple(start_x, multiple)
        new_start_y = lower_multiple(start_y, multiple)
        new_end_x = upper_multiple(end_x, multiple)
        new_end_y = upper_multiple(end_y, multiple)

        return ((new_start_x, new_start_y), (new_end_x, new_end_y))

    @staticmethod
    def max_bbox(
        bbox1: Tuple[Tuple[float, float], Tuple[float, float]],
        bbox2: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Compute the maximum bounding box that contains both input bounding boxes.

        Args:
            bbox1 (Tuple[Tuple[float, float], Tuple[float, float]]): First bounding box.
            bbox2 (Tuple[Tuple[float, float], Tuple[float, float]]): Second bounding box.

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]: The maximum bounding box.
        """
        min_x = min(bbox1[0][0], bbox2[0][0])
        min_y = min(bbox1[0][1], bbox2[0][1])
        max_x = max(bbox1[1][0], bbox2[1][0])
        max_y = max(bbox1[1][1], bbox2[1][1])
        return ((min_x, min_y), (max_x, max_y))
