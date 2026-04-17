import torch
import torch.nn as nn
from typing import Callable, Optional


class BaseGate(nn.Module):
    """
    Base class for all gating mechanisms.

    A gate maps an input tensor (e.g., weights, activations, or scores)
    to a continuous gating tensor typically in [0, 1].

    This class also supports:
    - Learnable parameters (handled by subclasses)
    - Annealing schedules (e.g., temperature scheduling)

    Notes
    -----
    Subclasses must implement the `forward` method.

    The `step` method is used to update any dynamic internal state
    (e.g., temperature annealing) during training.

    Examples
    --------
    >>> gate = SomeGateSubclass(...)
    >>> output = gate(x)
    >>> gate.step(global_step)
    """

    def __init__(
        self,
        annealer: Optional[str] = None
    ):
        """
        Initialize the BaseGate.

        Parameters
        ----------
        annealer : Callable[[int], float], optional
            A function that takes the current step index and returns
            the updated value for a dynamic parameter (e.g., temperature).
            If None, no annealing is applied.
        """
        super().__init__()
        self.annealer = annealer
        self._annealing_initialized = False
        self.annealing_config = {}
        self._step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes gate values.
        All subclasses must implement this method.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (weights, activations, or scores)

        Returns
        -------
        torch.Tensor
            Probability values returned from the Gate (typically in [0, 1])
        
        Examples
        --------
        >>> gate = SomeGateSubclass(...)
        >>> TO:DO
        
        """
        raise NotImplementedError("Subclasses must implement forward() method.")

    def annealing_step(self, step: Optional[int] = None) -> None:
        """
        Update internal state (e.g., annealing temperature).

        Parameters
        ----------
        step : int, optional
            Current global step. If None, an internal counter is used.

        Notes
        -----
        This method should be called once per training iteration.
        Subclasses can override `_update_annealed_parameters` to define behavior.
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
        # Base does nothing
        pass
    
    def _apply_annealing(self, step: int):
        raise NotImplementedError("_apply_annealing must be implemented")

    def _update_annealed_parameters(self, value: float) -> None:
        """
        Hook for subclasses to update annealed parameters.

        Parameters
        ----------
        value : float
            Updated value for the annealed parameter

        Notes
        -----
        Subclasses should override this method if they use annealing.
        """
        # self.temperature = value
        raise NotImplementedError("_update_annealed_parameters must be implemented by subclasses using annealing.")   

    def get_state(self) -> dict:
        """
        Get internal state for logging/debugging.

        Returns
        -------
        dict
            Dictionary containing gate-specific state variables
        """
        return {
            "step": self._step
        }
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for forward pass (semantic clarity).

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.forward(x)

    def hard_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Convert soft gate probabilities into a hard binary mask.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        threshold : float
            Threshold for binarization

        Returns
        -------
        torch.Tensor
            Binary mask
        """
        probs = self.get_probabilities(x)
        return (probs >= threshold).float()
    
    def requires_step(self) -> bool:
        """
        Whether this gate requires step updates.
        """
        return self.annealer is not None
    
    def get_gate_parameters(self):
        """
        Returns learnable gate-specific parameters.

        Useful for:
        - logging
        - custom optimizers
        """
        return {
            name: param
            for name, param in self.named_parameters()
        }
    
    def configure(self, **kwargs):
        """
        External configuration hook (e.g., from training loop)
        """
        for k, v in kwargs.items():
            self.annealing_config[k] = v