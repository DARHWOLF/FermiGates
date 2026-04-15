import torch

from fermigates.models import FermiConvClassifier, FermiMLPClassifier, FermiTransformerClassifier


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


def test_model_utilities_temperature_and_sparsity():
    model = FermiMLPClassifier(input_dim=10, hidden_dims=(8,), num_classes=2)
    model.set_temperature(0.3)
    kept, total, frac = model.compute_sparsity(threshold=0.5)

    assert total > 0
    assert 0 <= kept <= total
    assert 0.0 <= frac <= 1.0
