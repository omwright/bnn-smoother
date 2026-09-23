"""
TAGI — Tractable Approximate Gaussian Inference (Goulet et al., 2021) — baseline.

Follows the equations of the paper and the reference MATLAB implementation
(``functions/tagi.m``, ``functions/act.m``).

Two properties distinguish TAGI:

1. **Every covariance is diagonal**, stored as a vector, so the hidden-state
   smoother sums scalar contributions ``Σ_j Jz² · Δσ²`` where a full-covariance
   smoother would form ``K Σ Kᵀ`` and keep the cross terms.
2. **Activations are locally linearized**: it propagates ``g(E[x])`` with
   Jacobian ``J = g'(μ)``, never the exact moment ``E[g(x)]``.

Layer convention ``z^(ℓ) = W^(ℓ)ᵀ a^(ℓ-1) + b^(ℓ)``, then ``a^(ℓ) = g(z^(ℓ))``.
Weight means and variances are ``(n_in, n_out)``; biases are separate
``(n_out,)`` parameters rather than a row of the weight matrix.

For binary classification, ``output_activation: sigmoid`` selects TAGI's native
scheme: one linear output unit, targets encoded ``{0,1} → {+1,-1}``, a Gaussian
likelihood with ``sigma_v``, and a probit readout at inference, which is what
makes ``predict`` return a probability.  The encoding lives on the network
because the runners call ``method.net.update(x, y)`` directly.

Config schema
-------------
    hidden_sizes       list[int]  [50]      hidden layer widths
    activation         str        relu      hidden activation
    output_activation  str        linear    `sigmoid` selects binary mode
    sigma_v            float      0.32      observation noise STD (not variance)
    factor4Wp          float      0.25      weight prior scale
    factor4Bp          float      0.01      bias prior variance
    process_noise      float|str  0.0       NON-STANDARD, see below
    process_noise_scale float     0.0       multiplier when process_noise="prior_scaled"
    epochs             int        1         passes in `fit` (unused when streaming)
    dtype              str        float64
    device             str        cpu (set by the experiment, not per method)

``weight_prior`` is not exposed: TAGI's prior (``0.25/n_in`` for ReLU) is part
of the method, and is 8x tighter than the ``he`` default of ``2/n_in``.

``process_noise`` is **not part of TAGI** and defaults to 0.0, the faithful
setting, and is only here for testing purposes.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from bnn import cpu_randn

from .base import BNNMethod, pop_dtype_device

# Probit readout sharpness from `dp.obs2class`.
_ALPHA = 3.0

# Variances are updated by adding a negative increment (Jw·Δσ²·Jw with Δσ² < 0),
# so round-off can push them just below zero.
_VAR_FLOOR = 1e-12

_LN2PI = math.log(2.0 * math.pi)

# Activations whose prior uses the fan-in rule rather than the fan-avg rule.
_FAN_IN_PRIOR = ("relu", "softplus", "sigmoid")


class _TAGI:
    """TAGI network with diagonal weight covariance.

    State, per layer ``i`` in ``0 .. n_l-2``:

        mu_w[i]  (n_in, n_out)     weight means
        var_w[i] (n_in, n_out)     weight variances (diagonal covariance)
        mu_b[i]  (n_out,)          bias means
        var_b[i] (n_out,)          bias variances
    """

    def __init__(
        self,
        layer_sizes: list[int],
        act_fns: list[str],
        *,
        sigma_v: float,
        factor4Wp: float,
        factor4Bp: float,
        process_noise: float | str = 0.0,
        process_noise_scale: float = 0.0,
        binary: bool = False,
        dtype: torch.dtype = torch.float64,
        device: torch.device = None,
    ):
        """Validate the architecture and draw the prior."""
        if len(act_fns) != len(layer_sizes) - 1:
            raise ValueError(
                f"act_fns has length {len(act_fns)}, expected {len(layer_sizes) - 1}"
            )
        if act_fns[-1] != "linear":
            raise ValueError(
                "TAGI observes the pre-activation output z^(L) directly, so the "
                f"output activation must be 'linear' (got {act_fns[-1]!r}). "
                "For binary classification pass binary=True, which applies TAGI's "
                "probit readout on top of a linear output."
            )
        if binary and layer_sizes[-1] != 1:
            raise ValueError(
                "TAGI's binary encoding uses a single output unit "
                f"(got output_dim={layer_sizes[-1]})"
            )

        self.layers = layer_sizes
        self.n_l = len(layer_sizes)
        self.act_fns = act_fns
        self.sigma_v = sigma_v
        self.factor4Wp = factor4Wp
        self.factor4Bp = factor4Bp
        self.binary = binary
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        self._prior_vars = self._resolve_prior_vars()
        self.mu_w, self.var_w, self.mu_b, self.var_b = self._init_params()
        self.q_per_layer = self._resolve_process_noise(process_noise, process_noise_scale)

    # ── Initialisation ────────────────────────────────────────────────────

    def _resolve_prior_vars(self) -> list[float]:
        """Per-layer weight prior variance (`tagi.initializeWeightBias`)."""
        prior_vars = []
        for i in range(self.n_l - 1):
            n_in, n_out = self.layers[i], self.layers[i + 1]
            if self.act_fns[i] in _FAN_IN_PRIOR:
                prior_vars.append(self.factor4Wp / n_in)
            else:
                prior_vars.append(self.factor4Wp * 2.0 / (n_in + n_out))
        return prior_vars

    def _init_params(self):
        """Draw the prior.

        Unlike Wagner's KBNN, the bias mean is sampled rather than zeroed, and
        the bias carries its own variance scale.
        """
        mu_w, var_w, mu_b, var_b = [], [], [], []
        for i in range(self.n_l - 1):
            n_in, n_out = self.layers[i], self.layers[i + 1]
            vw, vb = self._prior_vars[i], self.factor4Bp

            mu_w.append(
                cpu_randn(n_in, n_out, device=self.device, dtype=self.dtype) * math.sqrt(vw)
            )
            var_w.append(torch.full((n_in, n_out), vw, dtype=self.dtype, device=self.device))
            mu_b.append(cpu_randn(n_out, device=self.device, dtype=self.dtype) * math.sqrt(vb))
            var_b.append(torch.full((n_out,), vb, dtype=self.dtype, device=self.device))
        return mu_w, var_w, mu_b, var_b

    def _cast(self, t: Tensor) -> Tensor:
        """Move an incoming tensor onto this net's device and dtype."""
        return t.to(device=self.device, dtype=self.dtype)

    def _resolve_process_noise(self, process_noise, scale) -> list[float]:
        """Per-layer process noise.  NON-STANDARD -- see the module docstring.

        A float applies everywhere; ``"prior_scaled"`` gives layer i
        ``scale * prior_var[i]``.  Prefer the latter, since one absolute value
        is mis-scaled across layers whose prior variances differ by fan-in.
        """
        if isinstance(process_noise, str):
            if process_noise != "prior_scaled":
                raise ValueError(
                    f"process_noise must be a float or 'prior_scaled', "
                    f"got {process_noise!r}"
                )
            return [scale * pv for pv in self._prior_vars]
        return [float(process_noise)] * (self.n_l - 1)

    # ── Activations (local linearization) ─────────────────────────────────

    def _activate(self, name: str, mu_z: Tensor, var_z: Tensor):
        """Return ``(mu_a, var_a, J)`` for `act.meanA` / `act.covarianceSa`.

        TAGI evaluates the activation and its derivative at the *mean*:
        ``mu_a = g(mu_z)``, ``J = g'(mu_z)``, ``var_a = J · var_z · J``.
        """
        if name == "relu":
            J = (mu_z > 0).to(self.dtype)
            mu_a = torch.clamp(mu_z, min=0.0)
        elif name == "linear":
            J = torch.ones_like(mu_z)
            mu_a = mu_z
        elif name == "sigmoid":
            mu_a = torch.sigmoid(mu_z)
            J = mu_a * (1.0 - mu_a)
        else:
            raise ValueError(f"Unsupported activation {name!r}")
        return mu_a, J * var_z * J, J

    # ── Forward pass ──────────────────────────────────────────────────────

    def _forward(self, x: Tensor):
        """Propagate moments.  ``x`` is ``(n, input_dim)``.

        Returns ``(mu_z_out, var_z_out, cache)``.  Each cache entry holds the
        quantities the backward pass needs for that layer:

            mu_a_in   input activation mean  (x for layer 0)
            J_in      g'(mu_z) of the *previous* layer, i.e. the Jacobian of the
                      activation that produced ``mu_a_in``
            mu_z      pre-activation mean
            var_z     pre-activation variance

        ``J_in`` sits in the same entry as the activation it belongs to, which
        makes the backward pass's off-by-one unrepresentable.
        """
        mu_a = x
        var_a = torch.zeros_like(x)
        J = torch.zeros_like(x)          # never read: nothing below layer 0
        cache = []

        for i in range(self.n_l - 1):
            mu_z = mu_a @ self.mu_w[i] + self.mu_b[i]
            var_z = (
                (var_a + mu_a**2) @ self.var_w[i]
                + var_a @ (self.mu_w[i] ** 2)
                + self.var_b[i]
            )

            cache.append(
                {"mu_a_in": mu_a, "J_in": J, "mu_z": mu_z, "var_z": var_z}
            )
            mu_a, var_a, J = self._activate(self.act_fns[i], mu_z, var_z)

        # Output activation is linear, so the last z is the network output.
        return cache[-1]["mu_z"], cache[-1]["var_z"], cache

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive moments of ``z^(L)``.  ``X`` is ``(n, input_dim)``."""
        mu_z, var_z, _ = self._forward(self._cast(X))
        return mu_z, var_z

    # ── Backward pass (Kalman filter + RTS smoothing) ─────────────────────

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor) -> None:
        """Single-sample online update.

        Parameters
        ----------
        x : Tensor, shape (input_dim,)
        y : Tensor, shape (output_dim,)
            In binary mode, labels in ``{0, 1}``; encoded to ``{+1, -1}`` here.
        """
        x = self._cast(x)
        y = self._cast(y)
        if self.binary:
            y = 1.0 - 2.0 * y

        # Non-standard random-walk process model; see the module docstring.
        # It must run BEFORE the forward pass: positivity of the updated var_w
        # rests on var_w·mu_a^2 <= var_z, and inflating var_w afterwards would
        # break that bound and let var_w go negative.
        for i in range(self.n_l - 1):
            if self.q_per_layer[i] > 0.0:
                self.var_w[i] = self.var_w[i] + self.q_per_layer[i]
                self.var_b[i] = self.var_b[i] + self.q_per_layer[i]

        _, _, cache = self._forward(x.unsqueeze(0))
        for st in cache:                       # drop the batch dimension
            for k in st:
                st[k] = st[k].squeeze(0)

        self._backward(y, cache)

    def _backward(self, y: Tensor, cache: list[dict]) -> list[tuple[Tensor, Tensor]]:
        """Kalman observation update, then RTS smoothing back through layers.

        Mutates the parameters in place and returns each layer's ``(dmu,
        dvar)``, outermost first.  Deltas are carried directly rather than
        recovered by subtracting posteriors: late in training ``|dvar| <<
        var_z``, and that round trip loses most of the significant digits.
        """
        # Step 1 — observation update on z^(L) (`tagi.fowardHiddenStateUpdate`).
        out = cache[-1]
        gain = out["var_z"] / (out["var_z"] + self.sigma_v**2)
        dmu = gain * (y - out["mu_z"])
        dvar = -gain * out["var_z"]
        deltas = [(dmu, dvar)]

        # Step 2 — RTS smoothing backwards through the layers.
        for i in reversed(range(self.n_l - 1)):
            st = cache[i]
            # Clamp only the divisor; clamping the stored var_z would break the
            # var_z >= var_w·mu_a^2 bound that keeps var_w non-negative.
            inv_var_z = 1.0 / torch.clamp(st["var_z"], min=_VAR_FLOOR)

            # Hidden states FIRST: this gain is defined against the forward
            # pass's joint prior, so it must read the pre-update mu_w.
            #
            #   Cov(z^(l)_j, z^(l-1)_k) = J_k · var_z_k^(l-1) · mu_w_kj
            #
            # z_k feeds every unit of the next layer, hence the sum over j.
            # Summing scalar contributions is where the diagonal restriction
            # bites: it treats the innovations at each j as independent, where a
            # full-covariance smoother keeps the cross terms.  Preserve this.
            if i > 0:
                prev = cache[i - 1]
                Jz = (
                    (st["J_in"] * prev["var_z"]).unsqueeze(1)
                    * self.mu_w[i]
                    * inv_var_z.unsqueeze(0)
                )
                dmu_next = (Jz * dmu.unsqueeze(0)).sum(dim=1)
                dvar_next = (Jz**2 * dvar.unsqueeze(0)).sum(dim=1)
                # The sum over j is not bounded by var_z_prev, so
                # `var_z_prev + dvar_next` can go negative: the diagonal
                # approximation over-counts an innovation shared across units.
                # Deliberately NOT clamped -- that is a real property of TAGI's
                # representation, and suppressing it would strengthen the
                # baseline.  Safe because only the delta is carried and dvar
                # cannot flip sign; the floors below absorb the rest.

            # Parameters: each w_ji feeds exactly one z_j, so no sum over units.
            #   Cov(z_j, w_ji) = var_w_ji · mu_a_i
            Jw = self.var_w[i] * st["mu_a_in"].unsqueeze(1) * inv_var_z.unsqueeze(0)
            self.mu_w[i] = self.mu_w[i] + Jw * dmu.unsqueeze(0)
            self.var_w[i] = torch.clamp(
                self.var_w[i] + Jw**2 * dvar.unsqueeze(0), min=_VAR_FLOOR
            )

            #   Cov(z_j, b_j) = var_b_j
            Jb = self.var_b[i] * inv_var_z
            self.mu_b[i] = self.mu_b[i] + Jb * dmu
            self.var_b[i] = torch.clamp(
                self.var_b[i] + Jb**2 * dvar, min=_VAR_FLOOR
            )

            if i > 0:
                dmu, dvar = dmu_next, dvar_next
                deltas.append((dmu, dvar))

        return deltas

    # ── Training / inference ──────────────────────────────────────────────

    @torch.no_grad()
    def fit(self, X: Tensor, Y: Tensor, epochs: int = 1) -> list[float]:
        """Online training, one sample at a time.  Returns per-sample NLL.

        Samples are visited in order (no shuffling), matching TAGI's own
        ``batchSize = 1`` streaming protocol.
        """
        X = self._cast(X)
        Y = self._cast(Y)
        losses = []
        for _ in range(epochs):
            for n in range(X.shape[0]):
                losses.append(self._nll(X[n], Y[n]))
                self.update(X[n], Y[n])
        return losses

    def _nll(self, x: Tensor, y: Tensor) -> float:
        """Gaussian NLL of one observation under the current posterior."""
        mu_z, var_z, _ = self._forward(x.unsqueeze(0))
        target = 1.0 - 2.0 * y if self.binary else y
        var = var_z.squeeze(0) + self.sigma_v**2
        nll = 0.5 * (torch.log(var) + _LN2PI + (target - mu_z.squeeze(0)) ** 2 / var)
        return float(nll.sum().item())

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and variance, both ``(n, output_dim)``.

        Regression: ``z^(L)`` moments with observation noise folded in.
        Binary: TAGI's probit readout (`dp.obs2class`), returning ``p(y = 1)``.
        The class encoding is ``y=0 → +1``, so the reference's
        ``Phi(mu/·) = p(class 0)`` becomes ``p(y=1) = Phi(-mu/·)`` here.
        """
        mu_z, var_z = self.forward(X)
        if self.binary:
            denom = torch.sqrt((1.0 / _ALPHA) ** 2 + var_z)
            return torch.special.ndtr(-mu_z / denom), var_z
        return mu_z, var_z + self.sigma_v**2


class TAGIMethod(BNNMethod):
    """Adapter: ``_TAGI`` <-> ``BNNMethod``.

    There is deliberately no ``forward``: TAGI has no full predictive
    covariance, so the runners' ``hasattr(method, "forward")`` check must fail
    and the full NLL must report N/A, as it does for Wagner's KBNN.
    """

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        """Translate the experiment config into a _TAGI."""
        super().__init__(input_dim, output_dim, config)

        cfg = dict(config)
        self.epochs = cfg.pop("epochs", 1)

        hidden = cfg.pop("hidden_sizes", [50])
        activation = cfg.pop("activation", "relu")
        output_activation = cfg.pop("output_activation", "linear")
        sigma_v = cfg.pop("sigma_v", 0.32)
        factor4_wp = cfg.pop("factor4Wp", 0.25)
        factor4_bp = cfg.pop("factor4Bp", 0.01)
        process_noise = cfg.pop("process_noise", 0.0)
        process_noise_scale = cfg.pop("process_noise_scale", 0.0)
        dtype, device = pop_dtype_device(cfg)

        # `sigmoid` selects TAGI's native binary head; the network stays linear.
        binary = output_activation == "sigmoid"

        layer_sizes = [input_dim] + list(hidden) + [output_dim]
        act_fns = [activation] * len(hidden) + ["linear"]

        self._net = _TAGI(
            layer_sizes,
            act_fns,
            sigma_v=sigma_v,
            factor4Wp=factor4_wp,
            factor4Bp=factor4_bp,
            process_noise=process_noise,
            process_noise_scale=process_noise_scale,
            binary=binary,
            dtype=dtype,
            device=device,
        )

    # ── BNNMethod interface ───────────────────────────────────────────────

    def _fit(self, X: Tensor, Y: Tensor) -> dict:
        """Stream the data through the update once per epoch."""
        losses = self._net.fit(X, Y, epochs=self.epochs)
        return {"losses": losses}

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and marginal variance."""
        return self._net.predict(X)

    @property
    def net(self) -> _TAGI:
        """Direct access for the streaming runners' ``update_single``."""
        return self._net
