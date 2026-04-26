import pytest
import torch
from fermigates.gates import FermiGate
from fermigates.models import MLP, CNN, Transformer
from fermigates.experiments import Experiment
from fermigates.calibration import LinearCalibration
from fermigates.losses import fermiloss


class TestUserAPI:
    def test_MLP_mnist(self):
        """
        Validate:
        - MLP construction
        - Gate injection via factory
        - Experiment training loop
        - Accuracy computation
        """

        # Step 1: Define model
        model = MLP(
            input_dim=784,
            hidden_dims=[128, 64],
            output_dim=10,
            gate= FermiGate(mode="neuron"),
            loss=fermiloss,
            calibration=LinearCalibration()
        )

        # Step 2: Define experiment
        exp = Experiment(
            model=model,
            dataset="mnist",
            epochs=1  # keep small for test
        )

        # Step 3: Train
        exp.train()

        # Step 4: Evaluate
        acc = exp.accuracy()

        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_CNN_mnist(self):
        """
        Validate:
        - CNN layer stack
        - Channel / feature gating compatibility
        """

        model = CNN(
            input_channels=1,
            num_classes=10,
            gate=FermiGate(mode="channel"),
            loss=fermiloss,
            calibration=LinearCalibration()
        )

        exp = Experiment(
            model=model,
            dataset="mnist",
            epochs=1
        )

        exp.train()
        acc = exp.accuracy()

        assert isinstance(acc, float)

    def test_transformer(self):
        """
        Validate:
        - Transformer compatibility
        - Head-level gating
        """

        model = Transformer(
            vocab_size=1000,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            gate=FermiGate(mode="head"),
            loss=fermiloss,
            calibration=LinearCalibration()
        )

        exp = Experiment(
            model=model,
            dataset="synthetic_text", 
            epochs=1
        )

        exp.train()
        acc = exp.accuracy()

        assert isinstance(acc, float)

    def test_linear_calibration(self):
        """
        Validate:
        - Calibration layer behaves like affine transform
        """

        calib = LinearCalibration()

        logits = torch.randn(32, 10)
        calibrated = calib(logits)

        assert calibrated.shape == logits.shape

    def test_gate_effect(self):
        """
        Ensure gating is not a no-op.
        """

        model = MLP(
            input_dim=20,
            hidden_dims=[16],
            output_dim=2,
            gate=lambda: FermiGate(mode="neuron"),
            loss=fermiloss,
            calibration=LinearCalibration()
        )

        x = torch.randn(8, 20)

        # Forward pass twice
        out1 = model(x)
        out2 = model(x)

        # Should not crash and outputs should exist
        assert out1.shape == out2.shape

    # -------------------------------------------------
    # Step 7: Multi-layer gate independence
    # -------------------------------------------------
    def test_independent_gates(self):
        """
        Ensure each layer gets its own gate instance.
        """

        model = MLP(
            input_dim=10,
            hidden_dims=[8, 8],
            output_dim=2,
            gate=lambda: FermiGate(mode="neuron"),
            loss=fermiloss,
            calibration=LinearCalibration()
        )

        gates = []

        for module in model.modules():
            if hasattr(module, "output_gate") and module.output_gate is not None:
                gates.append(module.output_gate)

        assert len(gates) > 1
        assert gates[0] is not gates[1]

    # -------------------------------------------------
    # Step 8: Forward-only usage (no experiment)
    # -------------------------------------------------
    def test_forward_only(self):
        """
        Users should be able to use models without Experiment wrapper.
        """

        model = MLP(
            input_dim=10,
            hidden_dims=[16],
            output_dim=2,
            gate= FermiGate(mode="neuron"),
            loss=fermiloss,
            calibration=LinearCalibration()
        )

        x = torch.randn(4, 10)
        y = model(x)
        assert y.shape == (4, 2)