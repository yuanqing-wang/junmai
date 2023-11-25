import torch

def test_exp_smearing():
    from dubonnet.layers import ExpNormalSmearing
    smearing = ExpNormalSmearing()
    dist = torch.linspace(0, 10, 100).unsqueeze(-1)
    out = smearing(dist)
    assert out.shape == (100, 50)
    assert out[-1, 0].item() == 0.0
