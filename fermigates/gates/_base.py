import torch
import torch.nn as nn
from typing import Callable, Optional


class BaseGate(nn.Module):
    """
    Abstract base class for all gating mechanisms in FermiGates.

    Gates are modules that map an input tensor (such as weights, activations, or scores)
    to a continuous gating tensor, typically with values in [0, 1].

    This class provides a unified interface for:
        - Defining learnable gating parameters (to be implemented by subclasses)
        - Supporting annealing schedules (e.g., temperature or slope scheduling)
        - Managing internal state for annealing and training
        - Converting soft gate outputs to hard binary masks

    Subclasses must implement the `forward` method to define the gating transformation.
    If annealing is used, subclasses should also implement `_apply_annealing` and
    `_update_annealed_parameters` to update dynamic parameters (such as temperature).

    Usage Example
    -------------
    >>> gate = SomeGateSubclass(...)
    >>> output = gate(x)  # Forward pass
    >>> gate.annealing_step(global_step)  # Update annealed parameters during training

    Notes
    -----
    - The `annealing_step` method should be called once per training iteration if annealing is used.
    - The `hard_mask` method can be used to obtain a binary mask from the gate's output probabilities.
    - The `get_gate_parameters` method returns all learnable parameters for logging or custom optimization.
    """

    def __init__(
        self,
        annealer: Optional[str] = None
    ):
        """
        Initialize the BaseGate.

        Parameters
        ----------
        annealer : Optional[Callable[[int], float]]
            A callable that takes the current step index and returns the updated value
            for a dynamic parameter (e.g., temperature or slope). If None, no annealing is applied.
        """
        super().__init__()
        self.annealer = annealer
        self._annealing_initialized = False
        self.annealing_config = {}
        self._step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gate output probabilities for the given input tensor.

        This method must be implemented by all subclasses to define the gating transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (e.g., weights, activations, or scores).

        Returns
        -------
        torch.Tensor
            Output tensor of probabilities, typically in the range [0, 1].

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement forward() method.")

    def annealing_step(self, step: Optional[int] = None) -> None:
        """
        Update the internal state for annealing (e.g., temperature or slope).

        This method should be called once per training iteration if annealing is used.
        It updates the step counter and applies the annealing schedule if provided.

        Parameters
        ----------
        step : Optional[int]
            The current global step. If None, an internal counter is incremented automatically.

        Notes
        -----
        Subclasses can override `_update_annealed_parameters` to define custom annealing behavior.
        """
        # Step 1: Update internal step counter
        if step is None:
            self._step += 1
            current_step = self._step
        else:
            self._step = step
            current_step = step

        # Step 2: Initialize annealing once
        if self.annealer is not None and not self._annealing_initialized:
            self._initialize_annealing()
            self._annealing_initialized = True

        # Step 3: Apply annealing
        if self.annealer is not None:
            self._apply_annealing(current_step)

    def _initialize_annealing(self):
        """
        Initialize any state required for annealing.

        This method is a no-op in the base class, but can be overridden by subclasses
        to set up initial values for annealed parameters.
        """
        pass

    def _apply_annealing(self, step: int):
        """
        Apply the annealing schedule for the current step.

        This method must be implemented by subclasses that use annealing.

        Parameters
        ----------
        step : int
            The current global step.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass using annealing.
        """
        raise NotImplementedError("_apply_annealing must be implemented by subclasses using annealing.")

    def _update_annealed_parameters(self, value: float) -> None:
        """
        Update the value of the annealed parameter (e.g., temperature or slope).

        This method should be overridden by subclasses that use annealing to update
        their internal parameters.

        Parameters
        ----------
        value : float
            The updated value for the annealed parameter.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass using annealing.
        """
        raise NotImplementedError("_update_annealed_parameters must be implemented by subclasses using annealing.")

    def get_state(self) -> dict:
        """
        Return the internal state of the gate for logging or debugging purposes.

        Returns
        -------
        dict
            Dictionary containing gate-specific state variables (e.g., current step).
        """
        return {
            "step": self._step
        }
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gate output probabilities for the given input tensor.

        This is an alias for the forward pass, provided for semantic clarity.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor of probabilities.
        """
        return self.forward(x)

    def hard_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Convert soft gate probabilities into a hard binary mask using a threshold.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the gate.
        threshold : float, default=0.5
            Threshold for binarization. Values greater than or equal to the threshold are set to 1, others to 0.

        Returns
        -------
        torch.Tensor
            Binary mask tensor (same shape as input).
        """
        probs = self.get_probabilities(x)
        return (probs >= threshold).float()
    
    def requires_step(self) -> bool:
        """
        Return whether this gate requires step updates (i.e., uses annealing).

        Returns
        -------
        bool
            True if the gate uses an annealer, False otherwise.
        """
        return self.annealer is not None
    
    def get_gate_parameters(self):
        """
        Return all learnable gate-specific parameters as a dictionary.

        Useful for logging, custom optimizers, or parameter inspection.

        Returns
        -------
        dict
            Dictionary mapping parameter names to parameter tensors.
        """
        return {
            name: param
            for name, param in self.named_parameters()
        }

    def configure(self, **kwargs):
        """
        External configuration hook for updating gate settings (e.g., from a training loop).

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments to update the gate's configuration (e.g., annealing parameters).
        """
        for k, v in kwargs.items():
            self.annealing_config[k] = v