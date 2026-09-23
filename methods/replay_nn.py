"""
Replay-buffer NN baseline — AdamW on a FIFO of recent samples.

The deterministic baseline of the VLA experiment.  One optimiser step per
arriving pair, on a minibatch drawn from a bounded FIFO, with the optimiser and
buffer persisting across the stream.

This is **not** :mod:`methods.sgd`, which updates on the arriving sample alone
with no buffer.  Under a moving distribution that matters: a FIFO as long as
the regime is never free of stale targets, and that is part of what the
comparison measures.  The two share the legend label "SGD", so the method name
is what distinguishes them; say in prose which one a figure uses.

Like ``sgd`` it has no calibrated uncertainty, so ``predict`` returns NaN
variances.

Layer names (``fc1``/``fc2``/``fc3``), the default ``nn.Linear`` init, buffer
semantics and sampling order are preserved from the archived experiment, which
produced the adapters this is checked against.  Changing them changes the
trained model, so do not "tidy" them.

Config keys
-----------
hidden_sizes      list[int]   hidden layer widths               [7, 7]
activation        str         hidden activation                 "relu"
output_activation str         output activation ("linear")      "linear"
optimizer         str         only "adamw" is implemented       "adamw"
lr                float       learning rate                     1e-3
betas             list[float] AdamW betas                       [0.9, 0.999]
eps               float       AdamW epsilon                     1e-8
weight_decay      float       AdamW L2 penalty                  0.0
buffer_size       int         FIFO capacity                     60
batch_size        int         samples drawn per update          16
updates_per_pair  int         optimiser steps per arrival       1
batch_seed        int         seed for the sampling generator   None
zero_init_output  bool        start at exactly zero residual    True
epochs            int         passes for batch ``.fit()``       1
dtype             str         "float32" or "float64"            "float32"
device            str         set by the experiment             "cpu"
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BNNMethod, pop_dtype_device

# =============================================================================
# FIFO replay buffer
# =============================================================================

class FIFOReplay:
    """Bounded FIFO with deterministic uniform sampling without replacement.

    Contents stay on the CPU whatever the net's device -- a CPU permutation
    cannot index device tensors -- and ``sample`` moves the minibatch, so the
    generator stream is identical on any device.
    """

    def __init__(self, capacity: int):
        """Create an empty buffer holding at most ``capacity`` pairs."""
        if capacity <= 0:
            raise ValueError(f"Replay capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self._items: deque[tuple[Tensor, Tensor]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        """Number of pairs currently buffered."""
        return len(self._items)

    def add(self, x: Tensor, y: Tensor) -> None:
        """Append one pair, evicting the oldest when full."""
        self._items.append((x.detach().cpu().clone(), y.detach().cpu().clone()))

    def sample(
        self, batch_size: int, generator: torch.Generator, device: torch.device = None
    ) -> tuple[Tensor, Tensor]:
        """Draw ``min(batch_size, len(self))`` distinct pairs.

        The newest sample is not guaranteed to appear: it competes with the
        rest of the buffer on equal terms, as in the archived experiment.
        """
        if not self._items:
            raise RuntimeError("Cannot sample an empty replay buffer")
        count = min(int(batch_size), len(self._items))
        indices = torch.randperm(len(self._items), generator=generator)[:count].tolist()
        x = torch.stack([self._items[i][0] for i in indices])
        y = torch.stack([self._items[i][1] for i in indices])
        if device is not None:
            x, y = x.to(device), y.to(device)
        return x, y

    def state_dict(self) -> dict[str, Any]:
        """Buffer contents as stacked (n, dim) tensors."""
        if self._items:
            x = torch.stack([item[0] for item in self._items])
            y = torch.stack([item[1] for item in self._items])
        else:
            x = y = torch.empty((0, 0))
        return {"capacity": self.capacity, "x": x, "y": y}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore buffer contents from :meth:`state_dict`."""
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"Replay capacity mismatch: {state['capacity']} != {self.capacity}"
            )
        x, y = state["x"], state["y"]
        if x.shape != y.shape or x.ndim != 2:
            raise ValueError(f"Invalid replay tensors: {tuple(x.shape)}, {tuple(y.shape)}")
        self._items.clear()
        for i in range(x.shape[0]):
            self.add(x[i], y[i])


# =============================================================================
# Inner network
# =============================================================================

class _ReplayNet(nn.Module):
    """Residual MLP with an online ``update(x, y)`` and a replay buffer.

    Layer names are load-bearing; see the module docstring.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        hidden_activation: str = "relu",
        output_activation: str = "linear",
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        buffer_size: int = 60,
        batch_size: int = 16,
        updates_per_pair: int = 1,
        batch_seed: int = None,
        zero_init_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        """Build the three layers, the optimiser and the replay buffer."""
        super().__init__()
        if len(layer_sizes) != 4:
            raise ValueError(
                f"replay_nn expects two hidden layers (four sizes), got {layer_sizes}"
            )
        if output_activation != "linear":
            raise ValueError(
                f"replay_nn supports only a linear output, got {output_activation!r}"
            )
        _acts = {"relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
        if hidden_activation not in _acts:
            raise ValueError(
                f"Unknown hidden activation: {hidden_activation!r}. "
                f"Choose from {sorted(_acts)}."
            )

        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.output_activation = output_activation
        self.batch_size = int(batch_size)
        self.updates_per_pair = int(updates_per_pair)
        self._hidden_act = _acts[hidden_activation]

        # Names and init preserved from the archived model.
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        if zero_init_output:
            # Exactly zero residual, so an untrained adapter is a no-op.
            nn.init.zeros_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)

        # nn.init ran on the CPU RNG, so initialisation matches across devices.
        self.to(device=self.device, dtype=dtype)

        # Built after .to() so the optimiser sees final dtypes.
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, betas=tuple(betas), eps=eps,
            weight_decay=weight_decay,
        )
        self.replay = FIFOReplay(buffer_size)
        # Separate from the global RNG, so minibatch draws neither depend on
        # nor perturb whatever else uses the default generator.
        self.batch_generator = torch.Generator()
        if batch_seed is not None:
            self.batch_generator.manual_seed(int(batch_seed))

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """Return the residual prediction for ``x``."""
        h = x.to(device=self.device, dtype=self.dtype)
        h = self._hidden_act(self.fc1(h))
        h = self._hidden_act(self.fc2(h))
        return self.fc3(h)

    # ── Online update ────────────────────────────────────────────────

    @torch.enable_grad()
    def update(self, x: Tensor, y: Tensor) -> float:
        """Buffer one pair, then take ``updates_per_pair`` minibatch steps.

        ``@torch.enable_grad()`` so it works inside the streaming loops' outer
        ``@torch.no_grad()``.
        """
        self.replay.add(x.to(dtype=self.dtype), y.to(dtype=self.dtype))
        loss_value = float("nan")
        for _ in range(self.updates_per_pair):
            bx, by = self.replay.sample(self.batch_size, self.batch_generator, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(self(bx), by)
            loss.backward()
            self.optimizer.step()
            loss_value = loss.item()
        return loss_value

    # ── Prediction ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Point prediction; variance is NaN (no calibrated uncertainty)."""
        mean = self(X)
        return mean, torch.full_like(mean, float("nan"))

    # ── Full training state (for exact replay verification) ──────────

    def training_state(self) -> dict[str, Any]:
        """Model, optimiser, buffer and generator state, for exact replay."""
        return {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_state_dict": self.replay.state_dict(),
            "batch_generator_state": self.batch_generator.get_state(),
        }

    def load_training_state(self, state: dict[str, Any]) -> None:
        """Restore everything :meth:`training_state` saved."""
        self.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.replay.load_state_dict(state["replay_state_dict"])
        self.batch_generator.set_state(state["batch_generator_state"])


# =============================================================================
# BNNMethod adapter
# =============================================================================

class ReplayNNMethod(BNNMethod):
    """FIFO-replay AdamW baseline wrapped for the experiment runners."""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        """Translate the experiment config into a _ReplayNet."""
        super().__init__(input_dim, output_dim, config)

        cfg = dict(config)
        hidden = cfg.pop("hidden_sizes", [7, 7])
        activation = cfg.pop("activation", "relu")
        output_activation = cfg.pop("output_activation", "linear")
        optimizer = cfg.pop("optimizer", "adamw")
        if str(optimizer).lower() != "adamw":
            raise ValueError(
                f"replay_nn implements AdamW only, got optimizer={optimizer!r}. "
                f"Use `sgd` for the no-buffer single-sample baseline."
            )
        lr = cfg.pop("lr", 1e-3)
        betas = cfg.pop("betas", (0.9, 0.999))
        eps = cfg.pop("eps", 1e-8)
        weight_decay = cfg.pop("weight_decay", 0.0)
        buffer_size = cfg.pop("buffer_size", 60)
        batch_size = cfg.pop("batch_size", 16)
        updates_per_pair = cfg.pop("updates_per_pair", 1)
        batch_seed = cfg.pop("batch_seed", None)
        zero_init_output = cfg.pop("zero_init_output", True)
        self.epochs = cfg.pop("epochs", 1)
        dtype, device = pop_dtype_device(cfg, default_dtype="float32")

        self._net = _ReplayNet(
            [input_dim] + hidden + [output_dim],
            hidden_activation=activation,
            output_activation=output_activation,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            updates_per_pair=updates_per_pair,
            batch_seed=batch_seed,
            zero_init_output=zero_init_output,
            dtype=dtype,
            device=device,
        )

    # ── BNNMethod interface ──────────────────────────────────────────

    def _fit(self, X: Tensor, Y: Tensor) -> dict:
        """Stream the samples in order, once per epoch.

        Deliberately not shuffled: the buffer's contents depend on arrival
        order, so shuffling would change what the method is.
        """
        losses = []
        for _epoch in range(self.epochs):
            for i in range(X.shape[0]):
                losses.append(self._net.update(X[i], Y[i]))
        return {"losses": losses}

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean, with NaN variance."""
        return self._net.predict(X)

    @property
    def net(self) -> _ReplayNet:
        """The underlying network, for the runners' ``update_single``."""
        return self._net
