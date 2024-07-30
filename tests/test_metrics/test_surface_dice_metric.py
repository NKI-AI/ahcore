from pathlib import Path

import pytest
import torch

from ahcore.metrics.metrics import WSiSurfaceDiceMetric
from ahcore.utils.data import DataDescription, GridDescription


@pytest.fixture
def data_description() -> DataDescription:
    num_classes = 4
    index_map = {"class1": 1, "class2": 2, "class3": 3}
    data_dir = Path("data_dir")
    manifest_database_uri = "manifest_database_uri"
    manifest_name = "manifest_name"
    split_version = "split_version"
    annotations_dir = Path("annotations_dir")
    training_grid = GridDescription(mpp=1.0, tile_size=(256, 256), tile_overlap=(0, 0), output_tile_size=(256, 256))
    inference_grid = GridDescription(mpp=1.0, tile_size=(256, 256), tile_overlap=(0, 0), output_tile_size=(256, 256))
    return DataDescription(
        num_classes=num_classes,
        index_map=index_map,
        data_dir=data_dir,
        manifest_database_uri=manifest_database_uri,
        manifest_name=manifest_name,
        split_version=split_version,
        annotations_dir=annotations_dir,
        training_grid=training_grid,
        inference_grid=inference_grid,
    )


@pytest.fixture
def metric(data_description: DataDescription) -> WSiSurfaceDiceMetric:
    class_thresholds = [0.0, 1.0, 1.0, 1.0]
    return WSiSurfaceDiceMetric(data_description=data_description, class_thresholds=class_thresholds)


def get_batch() -> tuple[torch.Tensor, torch.Tensor, None, str]:
    predictions = torch.randn(1, 4, 256, 256).float()
    target = torch.zeros(1, 4, 256, 256).float()  # Mock target tensor
    target[0, 1, 128, 128] = 1.0  # Simulate class 1 presence
    target[0, 2, 64, 64] = 1.0  # Simulate class 2 presence
    target[0, 3, 10, 10] = 1.0  # Simulate class 3 presence
    roi = None
    wsi_name = "test_wsi"
    return predictions, target, roi, wsi_name


def test_process_batch(metric: WSiSurfaceDiceMetric) -> None:
    predictions, target, roi, wsi_name = get_batch()
    metric.process_batch(predictions, target, roi, wsi_name)
    metric.get_wsi_score(wsi_name)

    assert wsi_name in metric.wsis
    assert all("surface_dice" in metric.wsis[wsi_name][i] for i in range(metric._num_classes))


def test_get_average_score(metric: WSiSurfaceDiceMetric) -> None:
    predictions, target, roi, wsi_name = get_batch()
    metric.process_batch(predictions, target, roi, wsi_name)
    metric.get_wsi_score(wsi_name)

    average_scores = metric.get_average_score()

    assert isinstance(average_scores, dict)
    assert all(
        f"{metric.name}/surface_dice/{metric._label_to_class[idx]}" in average_scores
        for idx in range(metric._num_classes)
    )


def test_static_average_wsi_surface_dice(metric: WSiSurfaceDiceMetric) -> None:
    precomputed_output = [
        [{"wsi_surface_dice": {"class1": 0.8, "class2": 0.7, "class3": 0.9}}],
        [{"wsi_surface_dice": {"class1": 0.85, "class2": 0.75, "class3": 0.95}}],
    ]

    average_scores = WSiSurfaceDiceMetric.static_average_wsi_surface_dice(precomputed_output)

    assert isinstance(average_scores, dict)
    assert "wsi_surface_dice/class1" in average_scores
    assert "wsi_surface_dice/class2" in average_scores
    assert "wsi_surface_dice/class3" in average_scores


def test_reset(metric: WSiSurfaceDiceMetric) -> None:
    predictions, target, roi, wsi_name = get_batch()
    metric.process_batch(predictions, target, roi, wsi_name)
    metric.reset()

    assert metric.wsis == {}


def test_surface_dice_edge_cases(metric: WSiSurfaceDiceMetric) -> None:
    # Note: In this test, background is ignored.
    predictions = torch.zeros(1, 4, 256, 256).float()
    target = torch.zeros(1, 4, 256, 256).float()
    # Set targets and predictions in such a way that the boundaries are 1 pixel apart.
    # The surface dice should be 1.0 for class 1, as a shift by 1 pixel falls under the tolerance limit.
    predictions[0, 1, 0:128, 0:128] = 1.0
    target[0, 1, 1:129, 1:129] = 1.0
    # Test case where there is no overlap between the target and the prediction boundaries.
    predictions[0, 2, 129:256, 129:256] = 1.0
    target[0, 2, 0:64, 0:64] = 1.0

    metric.process_batch(predictions, target, None, "wsi_1")
    metric.get_wsi_score("wsi_1")

    scores = metric.get_average_score()

    assert scores[f"{metric.name}/surface_dice/class1"] == 1.0
    assert scores[f"{metric.name}/surface_dice/class2"] == 0.0
    # Monai returns nan for class 3 since there are neither targets nor predictions but we change it to 1.0.
    assert scores[f"{metric.name}/surface_dice/class3"] == 1.0
