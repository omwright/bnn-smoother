"""
BNN training methods — common interface + registry.

    from methods import build, BNNMethod

    method = build("kbnn_onestep", input_dim=8, output_dim=1, config={...})
    log    = method.fit(X_train, Y_train)
    mean, var = method.predict(X_test)

To add a new method: import the class and add it to METHODS.
Guard optional-dependency imports with try/except so that
``import methods`` always works even if e.g. GPyTorch isn't installed.
"""

import torch

from .base import BNNMethod
from .kbnn_onestep import KBNNOneStepMethod
from .replay_nn import ReplayNNMethod
from .sgd import SGDMethod
from .tagi import TAGIMethod
from .wagner_kbnn import WagnerKBNNMethod

METHODS: dict[str, type[BNNMethod]] = {
    "kbnn_onestep": KBNNOneStepMethod,
    "wagner_kbnn": WagnerKBNNMethod,
    "tagi": TAGIMethod,
    "replay_nn": ReplayNNMethod,
    "sgd": SGDMethod,
}

# ── Optional methods (require extra dependencies) ────────────────────────────

try:
    from .gp import GPMethod
    METHODS["gp"] = GPMethod
except ImportError:
    pass


def build(
    name: str,
    *,
    input_dim: int,
    output_dim: int,
    config: dict,
    device: torch.device = None,
) -> BNNMethod:
    """Construct a method instance by registry name.

    Parameters
    ----------
    name : str
        Key in ``METHODS`` (e.g. ``"kbnn_onestep"``, ``"sgd"``, ``"gp"``).
    input_dim, output_dim : int
        Data dimensions — methods use these to set up network architecture.
    config : dict
        Method-specific hyperparameters (from YAML ``model:`` block).
    device : torch.device, optional
        Experiment device, injected into ``config`` so the method's own
        ``device`` key picks it up.  One device serves the whole experiment; it
        is not a per-method setting, and any ``device`` already in ``config`` is
        overwritten.  ``None`` leaves ``config`` untouched, which is what the
        runners that have no device support rely on.
    """
    if name not in METHODS:
        raise KeyError(
            f"Unknown method {name!r}. Available: {sorted(METHODS)}"
        )
    if device is not None:
        config = {**config, "device": torch.device(device)}
    return METHODS[name](input_dim=input_dim, output_dim=output_dim, config=config)
