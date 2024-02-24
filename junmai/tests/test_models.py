import pytest
import torch

def test_model_forward():
    from junmai.models import JunmaiModel
    layer = JunmaiModel(32, 64, 4)
    h = torch.randn(10, 32)
    x = torch.randn(10, 3)
    out = layer(h, x)

def test_layer_equivariance(equivariance_test_utils):
    translation, rotation, reflection = equivariance_test_utils
    from junmai.models import JunmaiModel
    model = JunmaiModel(32, 64, 4)
    h = torch.randn(10, 32)
    x = torch.randn(10, 3)
    y = model(h, x)
    y_translation = model(h, translation(x))
    y_rotation = model(h, rotation(x))
    y_reflection = model(h, reflection(x))
    assert torch.allclose(y_translation, y)
    assert torch.allclose(y_rotation, y)
    assert torch.allclose(y_reflection, y)