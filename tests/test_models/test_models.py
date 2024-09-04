import pytest
import torch

from ahcore.models.MIL.ABmil import ABMIL
from ahcore.models.MIL.transmil import TransMIL


@pytest.fixture
def input_data(B: int = 16, N_tiles: int = 1000, feature_dim: int = 768) -> torch.Tensor:
    return torch.randn(B, N_tiles, feature_dim)


def test_ABmil_shape(input_data: torch.Tensor) -> None:
    model = ABMIL(in_features=768)
    output = model(input_data)
    assert output.shape == (16, 1)


def test_TransMIL_shape(input_data: torch.Tensor) -> None:
    model = TransMIL(in_features=768, n_classes=2)
    output = model(input_data)
    assert output.shape == (16, 2)
