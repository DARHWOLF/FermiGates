import torch 
import torch.nn as nn

from fermigates.masks.fermimask import FermiMask
from fermigates.layers.calibration.linear_calibration import LinearCalibration
from fermigates.layers.linear_layers.fermilayer import FermiGatedLinear 
from typing import Tuple, Callable, Optional

class BaseFermiModel(nn.Module):
    """
    Abstract base class for models that use Fermi-gating and need linear calibration utilities.
    Subclasses should:
      - define layers (e.g. self.fc1 = FermiGatedLinear(...))
      - implement forward(...) (can use apply_calibration when needed)
    This class provides helper functions for temperature control, mu init, sparsity calculation,
    and linear calibration attachment.
    """
    def __init__(self):
        super().__init__()
        # store calibration modules by name
        self.calibrations = nn.ModuleDict()

    def set_temperature(self, T_new: float):
        """
        Set temperature T across all FermiMask instances in this model.
        """
        for m in self.modules():
            if isinstance(m, FermiMask):
                m.set_temperature(T_new)

    def init_mu_from_weights(self, percentile: float = 0.5, per_layer_neuron: bool = False):
        """
        Initialize μ for all FermiGatedLinear layers in the model by using the percentile of |weights|.
        percentile: 0..1. per_layer_neuron: compute per-output-neuron value if True.
        """
        for name, module in self.named_modules():
            if isinstance(module, FermiGatedLinear):
                module.initialize_mu_from_weight_percentile(percentile, per_layer=per_layer_neuron)

    def compute_sparsity(self, threshold: float = 0.5) -> Tuple[int, int, float]:
        """
        Compute sparsity across all FermiGatedLinear layers.
        Returns: (kept, total, fraction_kept)
        """
        kept = 0
        total = 0
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, FermiGatedLinear):
                    W = module.linear.weight
                    P = module.mask(W)
                    total += P.numel()
                    kept += (P > threshold).sum().item()
        frac = float(kept) / float(total) if total > 0 else 0.0
        return kept, total, frac

    # ----- Linear calibration utilities -----
    @staticmethod
    def solve_ridge_cpu(X: torch.Tensor, E: torch.Tensor, lam: float = 1e-3, add_bias: bool = True
                       ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Solve ridge regression in closed form on CPU for numerical stability:
          min_W ||X W - E||_F^2 + lam ||W||_F^2
        X: (n, d_in), E: (n, d_out)
        Returns: W_hat (d_in, d_out), b_hat (d_out,) if add_bias else None
        """
        # move to cpu & float64 for stability
        Xc = X.detach().cpu().to(dtype=torch.float64)
        Ec = E.detach().cpu().to(dtype=torch.float64)
        n, d_in = Xc.shape
        _, d_out = Ec.shape

        if add_bias:
            ones = torch.ones((n, 1), dtype=Xc.dtype, device=Xc.device)
            X_aug = torch.cat([Xc, ones], dim=1)  # (n, d_in+1)
            d_aug = d_in + 1
            XtX = X_aug.T @ X_aug
            XtX += lam * torch.eye(d_aug, dtype=XtX.dtype, device=XtX.device)
            XTE = X_aug.T @ Ec
            # solve
            try:
                W_aug = torch.linalg.solve(XtX, XTE)
            except RuntimeError:
                W_aug = torch.pinverse(XtX) @ XTE
            W_hat = W_aug[:d_in, :].to(dtype=X.dtype)
            b_hat = W_aug[d_in, :].to(dtype=X.dtype)
            return W_hat, b_hat
        else:
            XtX = Xc.T @ Xc
            XtX += lam * torch.eye(d_in, dtype=XtX.dtype, device=XtX.device)
            XTE = Xc.T @ Ec
            try:
                W = torch.linalg.solve(XtX, XTE)
            except RuntimeError:
                W = torch.pinverse(XtX) @ XTE
            return W.to(dtype=X.dtype), None

    def calibrate_with_loader(self,
                              layer_input_fn: Callable[[], torch.Tensor],
                              original_layer_fn: Callable[[], torch.Tensor],
                              pruned_layer_fn: Callable[[], torch.Tensor],
                              calibration_loader: torch.utils.data.DataLoader,
                              lam: float = 1e-3,
                              add_bias: bool = True,
                              name: Optional[str] = None,
                              device: Optional[torch.device] = None) -> LinearCalibration:
        """
        Compute LinearCalibration by collecting (X, E) on calibration_loader then solving ridge.
        Arguments:
          - layer_input_fn: function that given batch X returns the flattened input used by the layer (batch, d_in)
                            (e.g. a wrapper that flattens images or extract activations)
          - original_layer_fn: function that given same batch returns original dense layer output (batch, d_out)
          - pruned_layer_fn: function that returns pruned layer output (batch, d_out)
          - calibration_loader: yields input batches (data, label) or inputs
          - lam: ridge penalty
          - add_bias: whether to solve for bias term
          - name: optional name under which to store the calibration module in self.calibrations
          - device: device to place resulting calibration (defaults to model device)
        Returns:
          - calibration module (registered under self.calibrations[name] if name provided)
        NOTES:
          - The three functions (layer_input_fn, original_layer_fn, pruned_layer_fn) must capture the same batch input internally.
          - Typical pattern: define a closure that uses the batch from the loader and returns X_flat and outputs.
        """
        device = device or next(self.parameters()).device
        X_list = []
        E_list = []
        self.eval()
        with torch.no_grad():
            for batch in calibration_loader:
                # accommodate dataloader returning (inputs, labels) or raw inputs
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)

                # user-provided closures are expected to read 'inputs' and return tensors
                X_flat = layer_input_fn(inputs)           # (b, d_in)
                f_orig = original_layer_fn(inputs)        # (b, d_out)
                f_pruned = pruned_layer_fn(inputs)        # (b, d_out)

                # residual E = f_orig - f_pruned
                E = (f_orig - f_pruned).detach().cpu()
                X_list.append(X_flat.detach().cpu())
                E_list.append(E)

        if len(X_list) == 0:
            raise ValueError("Calibration loader produced no data.")

        X = torch.cat(X_list, dim=0)
        E = torch.cat(E_list, dim=0)

        W_hat, b_hat = self.solve_ridge_cpu(X, E, lam=lam, add_bias=add_bias)
        # create calibration module and load weights
        d_in = X.shape[1]
        d_out = E.shape[1]
        calib = LinearCalibration(d_in, d_out, learnable=False, device=device)
        calib.load_calibration(W_hat.to(device), b_hat.to(device) if b_hat is not None else None)
        if name:
            self.calibrations[name] = calib
        return calib

    def attach_calibration(self, name: str, calib: LinearCalibration):
        """
        Attach a pre-built calibration module under self.calibrations[name].
        """
        self.calibrations[name] = calib

    def apply_calibration(self, name: str, X_flat: torch.Tensor, y_pruned: torch.Tensor) -> torch.Tensor:
        """
        Apply stored calibration to produce corrected output:
          y_corrected = y_pruned + calib(X_flat)
        """
        if name not in self.calibrations:
            raise KeyError(f"Calibration '{name}' not found.")
        calib = self.calibrations[name]
        return y_pruned + calib(X_flat)

    # Optional: convenience helper to remove all calibrations
    def clear_calibrations(self):
        self.calibrations = nn.ModuleDict()
