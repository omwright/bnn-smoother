"""
Abstract base class for BNN training methods.

Every method wraps its internals behind this interface so that experiment
scripts can be method-agnostic:

    method = registry.build("kbnn_onestep", input_dim=8, output_dim=1, config={...})
    log = method.fit(X_train, Y_train)
    mean, var = method.predict(X_test)

``predict`` returns marginal (mean, variance), the lowest common denominator
across methods; the gradient-trained baselines report no variance at all.
Richer output goes through a method's own public methods, for callers that
know the concrete type.

By convention ``forward(x) -> (mean, cov)`` returns the full *predictive*
covariance for a single input, aleatoric noise included, so that ``diag(cov)``
matches ``predict``.  The runners score the full-covariance NLL against it
beside a diagonal NLL from ``predict``; if the two distributions disagree, the
NLLs are not comparable.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import torch
from torch import Tensor

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def pop_dtype_device(
    cfg: dict, *, default_dtype: str = "float64",
) -> tuple[torch.dtype, torch.device]:
    """Pop the shared ``dtype`` / ``device`` keys from a method config dict.

    :func:`methods.build` injects ``device`` from the one experiment-level
    setting; it is not a per-method knob.  It defaults to CPU.
    """
    dtype = _DTYPES[cfg.pop("dtype", default_dtype)]
    device = torch.device(cfg.pop("device", "cpu"))
    return dtype, device


class BNNMethod(ABC):
    """Unified interface for BNN training methods."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: dict,
    ):
        """Store the data dimensions and the method's config dict."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit(self, X: Tensor, Y: Tensor) -> dict:
        """Train on data.  Return a log dict (losses, diagnostics, ...)."""

    @abstractmethod
    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and marginal variance.

        X is (n, input_dim); both returned tensors are (n, output_dim).
        """

    # ------------------------------------------------------------------
    # Public entry point (wraps _fit with timing)
    # ------------------------------------------------------------------

    def fit(self, X: Tensor, Y: Tensor) -> dict:
        """Train and return log dict.  Always includes ``train_time``."""
        t0 = time.perf_counter()
        log = self._fit(X, Y)
        log.setdefault("train_time", time.perf_counter() - t0)
        return log
