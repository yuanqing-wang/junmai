import junmai
import torch

def test_layer_forward():
    from junmai.layers import JunmaiLayer
    layer = JunmaiLayer(32, 64)
    h = torch.randn(10, 32)
    x = torch.randn(10, 3)
    out = layer(h, x)