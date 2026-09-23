"""
Online SGD baseline — an MLP trained with single-sample AdamW updates.

A point-estimate baseline: it reports RMSE and accuracy/BCE but no calibrated
variance, so NLL-based metrics report NaN.  Each sample is seen exactly once,
in order, with one gradient step -- the same regime the Kalman methods run in,
with no replay buffer and no mini-batching.

Config keys
-----------
hidden_sizes      list[int]   hidden layer widths               [50]
activation        str         hidden activation                 "relu"
output_activation str         output activation                 "linear"
                              "linear"  → MSE loss
                              "sigmoid" → BCE loss
lr                float       Adam learning rate                3e-4
weight_decay      float       Adam L2 penalty                  1e-4
grad_clip         float       max gradient norm (0 to disable) 1.0
epochs            int         passes for batch .fit()           1
dtype             str         "float32" or "float64"           "float64"
device            str         set by the experiment            "cpu"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BNNMethod, pop_dtype_device

# =============================================================================
# Inner network
# =============================================================================

class _SGDNet(nn.Module):
    """MLP with its own optimiser and an online ``update(x, y)``, as the
    Kalman networks have."""

    def __init__(
        self,
        layer_sizes: list[int],
        hidden_activation: str = "relu",
        output_activation: str = "linear",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device = None,
    ):
        """Build the layers, initialise them, and create the optimiser."""
        super().__init__()
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.output_activation = output_activation
        self.grad_clip = grad_clip

        # ── Hidden activation function ───────────────────────────────
        _acts = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
        }
        if hidden_activation not in _acts:
            raise ValueError(
                f"Unknown hidden activation: {hidden_activation!r}.  "
                f"Choose from {sorted(_acts)}."
            )
        self._hidden_act = _acts[hidden_activation]
        self._hidden_act_name = hidden_activation

        # ── Build layers ─────────────────────────────────────────────
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.linear_layers.append(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )

        # ── Initialisation ───────────────────────────────────────────
        # Kaiming (He) normal on the hidden layers, to match the BNN priors;
        # Xavier on the output layer.
        nonlin = (
            "relu" if hidden_activation == "relu" else "linear"
        )
        for layer in self.linear_layers[:-1]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlin)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.linear_layers[-1].weight)
        nn.init.zeros_(self.linear_layers[-1].bias)

        # nn.init ran on the CPU RNG, so initialisation matches across devices.
        self.to(device=self.device, dtype=dtype)

        # After .to(), so the parameters already have the right dtype.
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay,
        )

    # ── Forward pass ─────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """Return logits.  A sigmoid output is applied later, in ``predict``
        and inside ``binary_cross_entropy_with_logits``."""
        h = x.to(device=self.device, dtype=self.dtype)
        for layer in self.linear_layers[:-1]:
            h = self._hidden_act(layer(h))
        return self.linear_layers[-1](h)

    # ── Loss ─────────────────────────────────────────────────────────

    def _compute_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        """BCE-with-logits for a sigmoid output, MSE for a linear one."""
        if self.output_activation == "sigmoid":
            return F.binary_cross_entropy_with_logits(logits, y)
        return F.mse_loss(logits, y)

    # ── Online update ────────────────────────────────────────────────

    @torch.enable_grad()
    def update(self, x: Tensor, y: Tensor) -> float:
        """One Adam step on a single sample; returns its loss.

        ``@torch.enable_grad()`` so it works inside the streaming loop's outer
        ``@torch.no_grad()``.  x is (input_dim,), y is (output_dim,).
        """
        self.train()

        # (1, dim) mini-batches, detached from any external graph.
        x_batch = x.detach().to(device=self.device, dtype=self.dtype).unsqueeze(0)
        y_batch = y.detach().to(device=self.device, dtype=self.dtype).unsqueeze(0)

        logits = self.forward(x_batch)
        loss = self._compute_loss(logits, y_batch)

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.optimizer.step()
        self.eval()

        return loss.item()

    # ── Batch prediction ─────────────────────────────────────────────

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Batch prediction.

        The mean is P(y=1|x) for a sigmoid output and the raw output
        otherwise.  Variance is NaN throughout: this is a point estimate, so
        NLL-based metrics correctly report NaN.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(X.to(device=self.device, dtype=self.dtype))
            if self.output_activation == "sigmoid":
                mean = torch.sigmoid(logits)
            else:
                mean = logits
        var = torch.full_like(mean, float("nan"))
        return mean, var


# =============================================================================
# BNNMethod adapter
# =============================================================================

class SGDMethod(BNNMethod):
    """Adapter: ``_SGDNet`` <-> ``BNNMethod``."""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        """Translate the experiment config into a _SGDNet."""
        super().__init__(input_dim, output_dim, config)

        cfg = dict(config)
        hidden = cfg.pop("hidden_sizes", [50])
        activation = cfg.pop("activation", "relu")
        output_activation = cfg.pop("output_activation", "linear")
        lr = cfg.pop("lr", 1e-3)
        weight_decay = cfg.pop("weight_decay", 1e-4)
        grad_clip = cfg.pop("grad_clip", 1.0)
        self.epochs = cfg.pop("epochs", 1)
        dtype, device = pop_dtype_device(cfg)

        layer_sizes = [input_dim] + hidden + [output_dim]

        self._net = _SGDNet(
            layer_sizes,
            hidden_activation=activation,
            output_activation=output_activation,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            dtype=dtype,
            device=device,
        )

    # ── BNNMethod interface ──────────────────────────────────────────

    def _fit(self, X: Tensor, Y: Tensor) -> dict:
        """Batch training: shuffle and update one sample at a time."""
        losses = []
        for _epoch in range(self.epochs):
            # Drawn on the CPU RNG deliberately: shuffling must not depend on
            # the device, and a CPU index tensor cannot index device tensors.
            perm = torch.randperm(X.shape[0]).tolist()
            for i in perm:
                loss = self._net.update(X[i], Y[i])
                losses.append(loss)
        return {"losses": losses}

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean, with NaN variance."""
        return self._net.predict(X)

    @property
    def net(self) -> _SGDNet:
        """The underlying network, for the runners' ``update_single``."""
        return self._net
