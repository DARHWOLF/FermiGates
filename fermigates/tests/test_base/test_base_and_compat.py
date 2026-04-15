import torch

from fermigates.base import BaseFermiClassifier
from fermigates.layers.linear_layers import FermiGatedLinear
from fermigates.masks import FermiMask, GompertzMask, GroupLassoMask, MagnitudeMask


class TinyClassifier(BaseFermiClassifier):
    def __init__(self):
        super().__init__(num_classes=2)
        self.fc = FermiGatedLinear(6, 2)

    def logits(self, x: torch.Tensor, *args, **kwargs):
        out, _ = self.fc(x)
        return out


def test_base_classifier_predict_and_sparsity():
    model = TinyClassifier()
    x = torch.randn(3, 6)

    logits = model(x)
    pred = model.predict(x)
    kept, total, frac = model.compute_sparsity()

    assert logits.shape == (3, 2)
    assert pred.shape == (3,)
    assert total > 0 and 0 <= kept <= total and 0.0 <= frac <= 1.0


def test_backward_compatible_mask_classes():
    fmask = FermiMask(shape=(4, 4))
    gmask = GompertzMask(size=4)
    lmask = GroupLassoMask(groups=2, group_size=2)
    mmask = MagnitudeMask(scores=torch.tensor([1.0, 2.0, 3.0]))

    w = torch.randn(4, 4)
    assert fmask(w).shape == (4, 4)
    assert gmask().shape == (4,)
    assert lmask().shape == (4,)
    assert mmask().shape == (3,)
