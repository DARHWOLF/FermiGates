import torch
from fermigates.masks.fermimask import FermiMask


def test_fermi_mask_output_range():
    mask = FermiMask(shape=(10, 10), init_mu=0.0, init_T=1.0)
    w = torch.randn(10, 10)

    P = mask(w)

    assert P.shape == w.shape
    assert torch.all(P >= 0.0)
    assert torch.all(P <= 1.0)


def test_fermi_mask_temperature_effect():
    w = torch.linspace(-1, 1, steps=100).view(10, 10)

    mask_hot = FermiMask(shape=w.shape, init_mu=0.0, init_T=10.0)
    mask_cold = FermiMask(shape=w.shape, init_mu=0.0, init_T=0.01)

    P_hot = mask_hot(w)
    P_cold = mask_cold(w)

    # Cold mask should be sharper (more values near 0 or 1)
    sharpness_hot = torch.std(P_hot)
    sharpness_cold = torch.std(P_cold)

    assert sharpness_cold > sharpness_hot


def test_set_temperature():
    mask = FermiMask(shape=(5, 5), init_mu=0.0, init_T=1.0)
    mask.set_temperature(0.25)

    assert torch.isclose(mask.T, torch.tensor(0.25))
