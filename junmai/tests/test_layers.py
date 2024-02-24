import junmai
import torch

def test_layer_forward():
    from junmai.layers import JunmaiLayer
    layer = JunmaiLayer(32, 64)
    h = torch.randn(10, 32)
    x = torch.randn(10, 3)
    out = layer(h, x)

def test_layer_equivariance(equivariance_test_utils):
    translation, rotation, reflection = equivariance_test_utils
    from junmai.layers import JunmaiLayer
    layer = JunmaiLayer(32, 64)
    h = torch.randn(10, 32)
    x = torch.randn(10, 3)
    y = layer(h, x)
    y_translation = layer(h, translation(x))
    y_rotation = layer(h, rotation(x))
    y_reflection = layer(h, reflection(x))
    assert torch.allclose(y_translation, y)
    assert torch.allclose(y_rotation, y)
    assert torch.allclose(y_reflection, y)
