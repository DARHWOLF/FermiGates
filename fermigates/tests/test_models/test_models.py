import torch
import torch.nn as nn

from fermigates.models import FermiConvClassifier, FermiMLPClassifier, FermiTransformerClassifier
from fermigates.models import CNN
from fermigates.models import Transformer


class _AddOneCalibration(nn.Module):
    """Simple calibration stub that shifts logits by +1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Return a deterministic shifted tensor for calibration checks.
        return x + 1.0


def test_fermi_mlp_classifier_logits_and_masks():
    model = FermiMLPClassifier(input_dim=32, hidden_dims=(16, 8), num_classes=4)
    x = torch.randn(5, 32)

    logits, masks = model.logits(x, return_masks=True)

    assert logits.shape == (5, 4)
    assert len(masks) == 3


def test_fermi_conv_classifier_logits_and_masks():
    model = FermiConvClassifier(in_channels=3, num_classes=10)
    x = torch.randn(2, 3, 32, 32)

    logits, masks = model.logits(x, return_masks=True)

    assert logits.shape == (2, 10)
    assert set(masks.keys()) == {"conv1", "conv2", "conv3", "classifier"}


def test_fermi_transformer_classifier_logits_shape():
    model = FermiTransformerClassifier(
        vocab_size=50,
        num_classes=3,
        max_seq_len=32,
        d_model=24,
        nhead=4,
        num_layers=2,
        dim_feedforward=48,
    )

    tokens = torch.randint(0, 50, (4, 12))
    attention_mask = torch.ones(4, 12, dtype=torch.long)

    logits, masks = model.logits(tokens, attention_mask=attention_mask, return_masks=True)

    assert logits.shape == (4, 3)
    assert "encoder" in masks and "head" in masks


def test_transformer_return_gate_outputs_and_calibration():
    # Step 1: Build two deterministic models with and without calibration.
    torch.manual_seed(23)
    model_plain = Transformer(
        vocab_size=64,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        num_classes=5,
        max_seq_len=64,
        dim_feedforward=32,
        gate=None,
        calibration=None,
    )
    torch.manual_seed(23)
    model_calibrated = Transformer(
        vocab_size=64,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        num_classes=5,
        max_seq_len=64,
        dim_feedforward=32,
        gate=None,
        calibration=_AddOneCalibration(),
    )

    # Step 2: Disable dropout randomness and run forward passes.
    model_plain.eval()
    model_calibrated.eval()
    tokens = torch.randint(0, 64, (3, 12))
    plain_logits = model_plain(tokens)
    calibrated_logits, gate_outputs = model_calibrated(tokens, return_gate_outputs=True)

    # Step 3: Validate calibration path and gate-output compatibility shape.
    assert torch.allclose(calibrated_logits, plain_logits + 1.0)
    assert isinstance(gate_outputs, list)
    assert len(gate_outputs) > 0


def test_cnn_return_gate_outputs_and_calibration():
    # Step 1: Build two deterministic CNN models with and without calibration.
    torch.manual_seed(41)
    model_plain = CNN(
        input_channels=1,
        num_classes=4,
        gate=None,
        calibration=None,
    )
    torch.manual_seed(41)
    model_calibrated = CNN(
        input_channels=1,
        num_classes=4,
        gate=None,
        calibration=_AddOneCalibration(),
    )

    # Step 2: Disable dropout randomness and run forward passes.
    model_plain.eval()
    model_calibrated.eval()
    x_value = torch.randn(3, 1, 28, 28)
    plain_logits = model_plain(x_value)
    calibrated_logits, gate_outputs = model_calibrated(x_value, return_gate_outputs=True)
    _, mask_dict = model_calibrated(x_value, return_masks=True)

    # Step 3: Validate calibration path and output payload compatibility.
    assert torch.allclose(calibrated_logits, plain_logits + 1.0)
    assert isinstance(gate_outputs, list)
    assert len(gate_outputs) == 4
    assert isinstance(mask_dict, dict)
    assert set(mask_dict.keys()) == {"conv1", "conv2", "conv3", "classifier"}


def test_model_utilities_temperature_and_sparsity():
    model = FermiMLPClassifier(input_dim=10, hidden_dims=(8,), num_classes=2)
    model.set_temperature(0.3)
    kept, total, frac = model.compute_sparsity(threshold=0.5)

    assert total > 0
    assert 0 <= kept <= total
    assert 0.0 <= frac <= 1.0
