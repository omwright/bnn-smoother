"""
The proposed method behind the BNNMethod interface.

A thin adapter: the smoothing logic lives in ``bnn/kalman_bnn_onestep.py``,
whose ``DEFAULTS`` lists every config key accepted here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bnn import KBNNOneStep

from .base import BNNMethod


class KBNNOneStepMethod(BNNMethod):
    """Adapter: KBNNOneStep ↔ BNNMethod."""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        """Translate the experiment config into a KBNNOneStep."""
        super().__init__(input_dim, output_dim, config)

        cfg = dict(config)
        self.epochs = cfg.pop("epochs", 1)

        hidden = cfg.pop("hidden_sizes", [50])
        cfg["layer_sizes"] = [input_dim] + hidden + [output_dim]

        self._net = KBNNOneStep(cfg)

    # -- BNNMethod interface -----------------------------------------------

    def _fit(self, X: Tensor, Y: Tensor) -> dict:
        """Stream the data through the smoother once per epoch."""
        losses = self._net.fit(X, Y, epochs=self.epochs)
        return {"losses": losses}

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and marginal variance."""
        return self._net.predict(X)

    # -- KBNN-specific extras -----------------------------------------------

    def forward(self, x: Tensor, **kw):
        """Full predictive covariance for a single input.

        Adds ``likelihood_noise`` to the latent covariance, so ``diag(cov)``
        equals what ``predict`` returns and the full NLL is scored against the
        same distribution as the diagonal one.  Extra outputs pass through.
        """
        mean, cov, *rest = self._net.forward(x, **kw)
        noise = self._net._likelihood_noise
        cov = cov + noise * torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
        return (mean, cov, *rest)

    @property
    def net(self) -> KBNNOneStep:
        """The underlying KBNNOneStep, for code that needs its internals."""
        return self._net
