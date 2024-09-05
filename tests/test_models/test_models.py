import pytest
import torch

from ahcore.models.layers.attention import GatedAttention, NystromAttention
from ahcore.models.layers.MLP import MLP
from ahcore.models.MIL.ABmil import ABMIL
from ahcore.models.MIL.transmil import TransMIL


@pytest.fixture
def input_data(B: int = 16, N_tiles: int = 1000, feature_dim: int = 768) -> torch.Tensor:
    return torch.randn(B, N_tiles, feature_dim)


def test_ABmil_shape(input_data: torch.Tensor) -> None:
    model = ABMIL(in_features=768)
    output = model(input_data)
    assert output.shape == (16, 1)

    output, attentions = model(input_data, return_attention_weights=True)
    assert output.shape == (16, 1)
    assert attentions.shape == (16, 1000, 1)


def test_TransMIL_shape(input_data: torch.Tensor) -> None:
    model = TransMIL(in_features=768, out_features=2)
    output = model(input_data)
    assert output.shape == (16, 2)


def test_MLP_shape(input_data: torch.Tensor) -> None:
    model = MLP(in_features=768, out_features=2, hidden=[128], dropout=[0.1])
    output = model(input_data)
    assert output.shape == (16, 1000, 2)


def test_MLP_hidden_dropout() -> None:
    with pytest.raises(ValueError):
        MLP(in_features=768, out_features=2, hidden=None, dropout=[0.1])


def test_attention_shape(input_data: torch.Tensor) -> None:
    model = GatedAttention(dim=768)
    output = model(input_data)
    assert output.shape == (16, 768)


def test_nystrom_att_with_mask(input_data: torch.Tensor) -> None:
    model = NystromAttention(
        dim=768, dim_head=768 // 8, heads=8, num_landmarks=1, pinv_iterations=6, residual=True, dropout=0.1
    )
    output, attn = model(input_data, mask=torch.ones_like(input_data, dtype=torch.bool)[:, :, 0], return_attn=True)
    assert output.shape == (16, 1000, 768)
    assert attn.shape == (16, 8, 1000, 1000)
