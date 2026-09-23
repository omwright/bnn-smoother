"""Experiment device resolution.

One device per experiment, set by the optional top-level ``device:`` key in a
YAML config.  ``methods.build`` injects the resolved device into each method's
config; the nets read it from their own ``device`` config key, as they already
did for ``dtype``.

Only CPU and CUDA are supported.
"""

from __future__ import annotations

import torch
from torch import Tensor

_SUPPORTED = ("cpu", "cuda")


def resolve_device(spec: str | torch.device | None = None) -> torch.device:
    """Resolve an experiment device specification.

    ``None`` or ``"auto"`` selects CUDA when available and CPU otherwise.
    Anything else is taken literally, so a config can pin ``cpu`` for
    reproducibility or name one GPU of several (``cuda:1``).
    """
    if spec is None or spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(spec)
    except RuntimeError as exc:  # torch rejects the string before we can inspect it
        raise ValueError(f"Unparseable device {str(spec)!r}: {exc}") from exc
    if device.type not in _SUPPORTED:
        raise ValueError(
            f"Unsupported device {str(spec)!r}: only {' and '.join(_SUPPORTED)} are supported. "
            f"MPS is not supported because the Kalman path needs linalg ops it lacks."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Device {str(spec)!r} was requested but CUDA is not available.")
    return device


def cpu_randn(*shape: int, device: torch.device = None, dtype: torch.dtype = torch.float64
              ) -> Tensor:
    """``torch.randn`` drawn on the CPU generator, then moved to ``device``.

    CUDA has its own generator, so drawing straight onto the GPU would give
    different weights from a CPU run at the same seed.  Drawing on the CPU keeps
    initialisation bit-identical across devices.
    """
    t = torch.randn(*shape, dtype=dtype)
    return t if device is None else t.to(device)
