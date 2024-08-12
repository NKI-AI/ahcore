import os
import pickle
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, Sequence
from warnings import warn

import dotenv
import imageio.v3 as iio
import numpy as np
from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.background import compute_masked_indices
from dlup.data.transforms import convert_annotations
from pymilvus import Collection
from sklearn.cluster import KMeans

from ahcore.utils.data import DataDescription
from ahcore.utils.manifest import DataManager
from ahcore.utils.vector_database.utils import connect_to_milvus, dict_to_uuid, load_data_description

log = getLogger(__name__)
dotenv.load_dotenv(override=True)


class ReduceMethod(Enum):
    MEAN = "MEAN"
    KMEANS5 = "KMEANS5"
    MEAN_KMEANS5 = "MEAN_KMEANS5"

    @staticmethod
    def from_value(value: str):
        """
        Converts a string value to a corresponding Enum value if possible.

        Args:
        value (str): The string representation of the Enum value.

        Returns:
        ReduceMethod: The corresponding Enum value.

        Raises:
        ValueError: If the input value does not correspond to any Enum value.
        """
        try:
            return ReduceMethod[value.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown reduce method {value}. Supported methods are {[method.value for method in ReduceMethod]}."
            )


class AnnotatedVectorQuerier:
    def __init__(
        self, collection_name: str, data_description: DataDescription, index_params: dict[str, Any] | None = None
    ):
        self.data_description = data_description
        self._vector_entry_name = "embedding"
        self._data_manager = DataManager(database_uri=data_description.manifest_database_uri)
        self._filenames: list[str] = self._determine_filenames()
        self._cache_folder = Path(os.environ.get("CACHE_FOLDER_VECTOR_LOOKUP"))
        self._cache_folder_annotation_masks = self._cache_folder / "annotation_masks"
        os.makedirs(self._cache_folder_annotation_masks, exist_ok=True)

        self.collection = Collection(name=collection_name, using=os.environ.get("MILVUS_ALIAS"))
        self._setup_index(index_params=index_params)
        self.collection.load()

    def _setup_index(self, index_params: dict[str, Any] | None = None) -> None:
        if index_params is None:
            index_params = {
                "index_type": "FLAT",
                "metric_type": "COSINE",
                "params": {},
            }
        self.collection.create_index(field_name=self._vector_entry_name, index_params=index_params)

    def _determine_filenames(self) -> list[str]:
        log.info("Determining filenames for annotated vector lookup.")

        def _extract_stem_from_filename(filename: str) -> str:
            return str(Path(filename).stem)

        patients = self._data_manager.get_records_by_split(
            manifest_name=self.data_description.manifest_name,
            split_version=self.data_description.split_version,
            split_category="predict",
        )
        filenames = [_extract_stem_from_filename(image.filename) for patient in patients for image in patient.images]
        return filenames

    @staticmethod
    def _convert_entries_to_sequence(entries: list[dict[str, Any]]) -> Sequence[tuple[float, float, int, int, float]]:
        converted_list: Sequence[tuple[float, float, int, int, float]] = [
            (entry["coordinate_x"], entry["coordinate_y"], entry["tile_size"], entry["tile_size"], entry["mpp"])
            for entry in entries
        ]
        return converted_list

    @staticmethod
    def _extract_embeddings(entries_dict: list[dict[str, Any]]) -> np.ndarray:
        embeddings = [entry["embedding"] for entries in entries_dict.values() for entry in entries]
        embeddings_np = np.array(embeddings)
        return embeddings_np

    @staticmethod
    def _find_kmeans_centroids(embeddings: np.ndarray, k: int = 5) -> np.ndarray:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        return kmeans.cluster_centers_

    def _convert_annotation_to_mask(
        self, filename: str, annotation: WsiAnnotations, region_size: tuple[int, int]
    ) -> np.ndarray:
        name_dict = self.data_description.model_dump()  # This includes the annotation class in roi_name
        name_dict["filename"] = filename
        name_dict["region_size"] = region_size
        cache_filename = self._cache_folder_annotation_masks / f"{dict_to_uuid(name_dict)}.tiff"

        if cache_filename.exists():
            annotation_mask = iio.imread(cache_filename)
        else:
            # expects region size of (height, width)
            _, annotation_mask, _ = convert_annotations(
                annotation,
                region_size=region_size,
                index_map=self.data_description.index_map,
            )
            annotation_mask = annotation_mask.astype(np.uint8)
            iio.imwrite(cache_filename, annotation_mask, compression="lzw")
        return annotation_mask

    def _prepare_slide_and_annotation(
        self, filename: str
    ) -> tuple[SlideImage, WsiAnnotations, tuple[int, int, int, int]]:
        slide_image = SlideImage.from_file_path(
            self.data_description.data_dir / f"{filename}.svs", internal_handler="vips"
        )
        annotations = WsiAnnotations.from_geojson(self.data_description.annotations_dir / f"{filename}.svs.geojson")
        annotations.filter(self.data_description.roi_name)
        bbox = annotations.bounding_box
        return slide_image, annotations, bbox

    def _query_entries_in_bbox(
        self, filename: str | Path, bbox: tuple[int, int, int, int], tile_size: int
    ) -> list[dict[str, Any]]:
        """Queries database for entries within a bounding box."""
        (x, y), (w, h) = bbox
        min_x, max_x, min_y, max_y = int(x), int(x + w - tile_size), int(y), int(y + h - tile_size)
        expr = (
            f"filename == '{str(filename)}.svs' and coordinate_x >= {min_x} and coordinate_x <= {max_x} "
            f"and coordinate_y >= {min_y} and coordinate_y <= {max_y}"
        )
        results = self.collection.query(
            expr=expr,
            output_fields=["*"],
        )

        return results

    def _find_entries_in_annotated_regions(
        self, threshold: float = 0.75, force_recompute: bool = False
    ) -> dict[str, list[dict[str, Any]]]:
        """Finds database entries and all embeddings in annotated regions for each filename in self._filenames."""
        save_dict = self.data_description.model_dump()
        save_dict["threshold"] = threshold
        cache_filename = self._cache_folder / f"{dict_to_uuid(save_dict)}.pkl"
        tile_size = self.data_description.inference_grid.tile_size[0]
        if cache_filename.exists() and not force_recompute:
            with open(cache_filename, "rb") as f:
                annotated_vector_dict = pickle.load(f)
        else:
            annotated_vector_dict = {}
            for filename in self._filenames:
                log.info(f"Processing {filename}")
                slide_image, annotations, annotation_bbox = self._prepare_slide_and_annotation(filename)
                entries_in_bbox = self._query_entries_in_bbox(
                    filename, annotation_bbox, tile_size
                )  # Queries via network to Milvus
                if not entries_in_bbox:
                    log.info(
                        f"No database entries found in bbox for {filename} for class {self.data_description.roi_name}."
                    )
                    continue

                regions = self._convert_entries_to_sequence(entries_in_bbox)
                region_size = slide_image.size[::-1]  # expects region size of (height, width)
                annotation_mask = self._convert_annotation_to_mask(
                    filename=filename, annotation=annotations, region_size=region_size
                )
                masked_indices = compute_masked_indices(
                    slide_image=slide_image, background_mask=annotation_mask, regions=regions, threshold=threshold
                )
                if masked_indices.size > 0:
                    annotated_vector_dict[filename] = [entries_in_bbox[i] for i in masked_indices]
                else:
                    warn(
                        f"No vector database entries found in annotated regions for {filename}  \
                        for class {self.data_description.roi_name}."
                    )
            with open(cache_filename, "wb") as f:
                pickle.dump(annotated_vector_dict, f)

        return annotated_vector_dict

    def get_reference_vectors(
        self, overlap_threshold: float, reduce_method: str = "mean", force_recompute: bool = False
    ) -> np.ndarray:
        """
        Computes reference vectors by aggregating embeddings from database entries that are within
        annotated regions of interest. This function first identifies database entries located within
        annotated regions based on a specified overlap threshold. It then extracts embeddings for these
        entries and aggregates them to form reference vectors, which can be used as templates or
        representative features for further analysis. The aggregation method used (e.g., mean) can be
        specified through the 'reduce_method' parameter.

        Args:
            overlap_threshold (float): The threshold for determining whether a database entry is
                                    considered within an annotated region. Entries with overlap above
                                    this threshold are included in the computation of reference vectors.
            reduce_method (str): The method used to aggregate embeddings into a single reference vector
                                per annotated region. Supported values are "mean". This method calculates
                                the mean across all embeddings.
            force_recompute (bool): If True, forces the re-computation of embeddings and reference vectors
                                    even if cached results exist. Default is False, where the method will
                                    first attempt to use cached results to avoid redundant computations.

        Returns:
            np.ndarray: An array of aggregated reference vectors, one for each region, based on the
                        specified reduction method. The shape and content of the array depend on the
                        dimensions of the embeddings and the chosen aggregation method.

        Raises:
            ValueError: If an unsupported value is provided to the `reduce_method` parameter.

        Example:
            >>> querier = AnnotatedVectorQuerier(collection, data_description)
            >>> reference_vectors = querier.get_reference_vectors(overlap_threshold=0.75, reduce_method="mean")
            >>> print(reference_vectors.shape)
        """
        reduce_method = ReduceMethod.from_value(reduce_method)
        entries = self._find_entries_in_annotated_regions(overlap_threshold, force_recompute)
        embeddings = self._extract_embeddings(entries)
        log.info(f"Found {len(embeddings)} vectors in annotated regions.")
        if reduce_method == ReduceMethod.MEAN:
            reference_vectors = np.mean(embeddings, axis=0)
            reference_vectors = reference_vectors.tolist()  # Convert to list for compatibility with Milvus
        elif reduce_method == ReduceMethod.KMEANS5:
            kmean_centroids = self._find_kmeans_centroids(embeddings, k=100)
            reference_vectors = kmean_centroids.tolist()
        elif reduce_method == ReduceMethod.MEAN_KMEANS5:
            mean_vector = np.mean(embeddings, axis=0)
            kmean_centroids = self._find_kmeans_centroids(embeddings, k=100)
            reference_vectors = np.vstack([mean_vector, kmean_centroids])
            reference_vectors = reference_vectors.tolist()
        else:
            raise NotImplementedError(f"Unsupported reduce method {reduce_method}.")
        return reference_vectors


if __name__ == "__main__":
    connect_to_milvus(
        host=os.environ.get("MILVUS_HOST"),
        user=os.environ.get("MILVUS_USER"),
        password=os.environ.get("MILVUS_PASSWORD"),
        port=os.environ.get("MILVUS_PORT"),
        alias=os.environ.get("MILVUS_ALIAS"),
    )
    collection = Collection(name="uni_collection_concat_train", using=os.environ.get("MILVUS_ALIAS"))
    data_description = load_data_description(os.environ.get("DATA_DESCRIPTION_PATH_ANNOTATED"))
    ann_querier = AnnotatedVectorQuerier(collection, data_description)
    result_dict = ann_querier._find_entries_in_annotated_regions(threshold=0.9)
