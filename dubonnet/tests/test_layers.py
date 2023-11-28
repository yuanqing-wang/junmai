import dubonnet
import torch

def test_exp_smearing():
    from dubonnet.layers import ExpNormalSmearing
    smearing = ExpNormalSmearing()
    dist = torch.linspace(0, 10, 100).unsqueeze(-1)
    out = smearing(dist)
    assert out.shape == (100, 50)
    assert out[-1, 0].item() == 0.0

def test_parameter_generation():
    from dubonnet.layers import ParameterGeneration
    h = torch.randint(5, size=(100,))
    h = torch.nn.functional.one_hot(h, 5).float()
    parameter_generation = ParameterGeneration(
        in_features=5,
        hidden_features=10,
        num_basis=128,
        num_rbf=50,
        num_heads=1,
    )
    K, Q, W0, W1 = parameter_generation(h)
    assert K.shape == (100, 100, 50, 128)
    assert Q.shape == (100, 100, 50, 128)
    assert W0.shape == (100, 128, 128)
    assert W1.shape == (100, 128, 1)

def test_basis_generation():
    from dubonnet.layers import BasisGeneration, ExpNormalSmearing
    smearing = ExpNormalSmearing()
    basis_generation = BasisGeneration(smearing)
    x = torch.randn(100, 3)
    basis = basis_generation(x)
    assert basis.shape == (100, 100, 3, 50)

def test_dubonnet_forward():
    from dubonnet.layers import (
        BasisGeneration, ExpNormalSmearing, ParameterGeneration
    )
    from dubonnet.models import DubonNet
    smearing = ExpNormalSmearing()
    basis_generation = BasisGeneration(smearing)
    dubonnet = DubonNet()
    parameter_generation = ParameterGeneration(
        in_features=5,
        hidden_features=10,
        num_rbf=50,
        num_basis=128,
        num_heads=1,
    )

    x = torch.randn(100, 3)
    h = torch.randint(5, size=(100,))
    h = torch.nn.functional.one_hot(h, 5).float()

    basis = basis_generation(x)
    K, Q, W0, W1 = parameter_generation(h)    
    Z = dubonnet(basis, (K, Q, W0, W1))
