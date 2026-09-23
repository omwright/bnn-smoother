"""
The KBNN of Wagner et al. (2023), arXiv:2110.00944 — baseline.

Assumes pairwise independence between neurons: the weight covariance is one
(n_in, n_in) matrix per output neuron, and activation covariances between
layers are diagonal.

The numerics follow the original source: the same variance decomposition
(A + B + C), moment formulas, per-neuron scalar Kalman gain and rank-1 weight
covariance update. What differs here is the wrapper and the expanded weight
prior initialization.

Config keys
-----------
hidden_sizes          list[int]   hidden layer widths                  [50]
activation            str         hidden activation ("relu", "sigmoid") "relu"
output_activation     str         output activation ("linear", etc.)   "linear"
use_bias              bool        include bias weights                 True
noise                 float       variance floor added at each layer   0.01
normalise             bool        divide by fan-in (Wagner's flag)     False
weight_prior          str         "scalar" or "he"                     "scalar"
weight_init_scale     float       mean std (scalar mode only)          1.0
weight_cov_init_scale float       cov diagonal (scalar) or He          1.0
                                  multiplier (he mode)
epochs                int         passes over the training set         1
dtype                 str         "float32" or "float64"               "float64"
device                str         set by the experiment                "cpu"

Weight prior modes
------------------
``"scalar"``   every layer uses ``weight_init_scale`` for the mean and
               ``weight_cov_init_scale`` for the covariance, whatever the
               fan-in.  Matches the original at 1.0 and 1.0.
``"he"``       per-weight prior variance ``weight_cov_init_scale * 2 / n_in``,
               which keeps the forward activation variance O(1) under
               standard-scaled inputs.  ``weight_init_scale`` is ignored.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions.normal import Normal

from bnn import cpu_randn

from .base import BNNMethod, pop_dtype_device

# ── Helpers ──────────────────────────────────────────────────────────────────

_PI = math.pi
_STD_NORMAL = Normal(0, 1)


def _gauss_pdf(x: Tensor, m: Tensor, C: Tensor) -> Tensor:
    """N(x; m, C) evaluated element-wise.  C is variance, not std."""
    return Normal(m, torch.sqrt(C)).log_prob(x).exp()


def _sigmoid(x: Tensor) -> Tensor:
    """Logistic sigmoid."""
    return 1 / (1 + torch.exp(-x))


# ── Core network ─────────────────────────────────────────────────────────────

class _WagnerBNN:
    """Internals of the Wagner et al. diagonal-covariance BNN.

    Weight layout per layer i:
        mw[i]  — (n_in, n_out)        weight means (last row = bias if use_bias)
        Cw[i]  — (n_out, n_in, n_in)  per-neuron weight covariance
    """

    def __init__(self, layer_sizes: list[int], act_fns: list[str], *,
                 use_bias: bool, noise: float, normalise: bool,
                 dtype: torch.dtype, device: torch.device = None,
                 weight_prior: str = "scalar",
                 weight_init_scale: float = 1.0,
                 weight_cov_init_scale: float = 1.0):
        """Build the per-neuron weight means and covariances."""
        self.layers = layer_sizes
        self.n_l = len(layer_sizes)
        self.act_fns = act_fns
        self.use_bias = use_bias
        self.noise = noise
        self.normalise = normalise
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.weight_prior = weight_prior
        self.weight_init_scale = weight_init_scale
        self.weight_cov_init_scale = weight_cov_init_scale

        if weight_prior not in ("scalar", "he"):
            raise ValueError(
                f"Unknown weight_prior {weight_prior!r}. "
                f"Choose 'scalar' (flat, backward compatible) or 'he' (per-layer He/NNGP)."
            )

        self.mw, self.Cw = self._init_weights()

    # ── Device-aware tensor factories ─────────────────────────────────────

    def _zeros(self, *shape) -> Tensor:
        """Zeros on this net's device and dtype."""
        return torch.zeros(*shape, dtype=self.dtype, device=self.device)

    def _ones(self, *shape) -> Tensor:
        """Ones on this net's device and dtype."""
        return torch.ones(*shape, dtype=self.dtype, device=self.device)

    # ── Initialisation ────────────────────────────────────────────────────

    @staticmethod
    def _prior_scales(mode: str, n_in: int,
                      weight_init_scale: float,
                      weight_cov_init_scale: float) -> tuple[float, float]:
        """Return (mean_std, cov_scale) for a single layer."""
        if mode == "he":
            prior_var = weight_cov_init_scale * 2.0 / n_in
            return (math.sqrt(prior_var), prior_var)
        else:
            return (weight_init_scale, weight_cov_init_scale)

    def _init_weights(self):
        """Draw weight means and set covariances from the configured prior."""
        mw = [None] * (self.n_l - 1)
        Cw = [None] * (self.n_l - 1)
        for i in range(self.n_l - 1):
            ni = self.layers[i] + (1 if self.use_bias else 0)
            no = self.layers[i + 1]

            mean_std, cov_scale = self._prior_scales(
                self.weight_prior, ni,
                self.weight_init_scale, self.weight_cov_init_scale,
            )

            mw[i] = cpu_randn(ni, no, device=self.device, dtype=self.dtype) * mean_std
            if self.use_bias:
                mw[i][-1] = 0.0

            Cw[i] = self._zeros(no, ni, ni)
            for j in range(no):
                Cw[i][j] = cov_scale * torch.eye(ni, dtype=self.dtype, device=self.device)
        return mw, Cw

    # ── Forward pass (moment propagation) ─────────────────────────────────

    def forward(self, x: Tensor, *, training: bool = False):
        """Propagate mean and diagonal variance through the network.

        x is (n_samples, input_dim).  With ``training``, returns the per-layer
        lists the backward pass needs instead of the final layer alone.
        """
        x = x.to(device=self.device, dtype=self.dtype)
        n_samples = x.shape[0]
        mz = x

        if self.use_bias:
            mz_ = torch.cat([x, self._ones(n_samples, 1)], dim=1)
        else:
            mz_ = x

        Cz_ = self._zeros(n_samples, mz_.shape[1], mz_.shape[1])

        all_ma, all_Ca, all_my, all_Cy = [], [], [], []

        for i in range(self.n_l - 1):
            ni = self.layers[i] + (1 if self.use_bias else 0)
            activation = self.act_fns[i]

            # ── Pre-activation mean ──────────────────────────────────
            ma = mz_.mm(self.mw[i])                                   # (n, no)

            # ── Pre-activation variance (A + B + C) ──────────────────
            #   A = diag(W^T Cz W)       input uncertainty
            #   B = mz^T Cw[j] mz        weight uncertainty (per neuron j)
            #   C = diag(Cz) . diag(Cw)  interaction
            A = torch.diagonal(
                torch.matmul(self.mw[i].T, torch.matmul(Cz_, self.mw[i])),
                dim1=1, dim2=2,
            )                                                          # (n, no)
            B = torch.einsum(
                'nmi,ni->nm',
                torch.einsum('mij,nj->nmi', self.Cw[i], mz_),
                mz_,
            )                                                          # (n, no)
            C = (
                torch.diagonal(Cz_, dim1=-2, dim2=-1)
                .mm(torch.diagonal(self.Cw[i], dim1=2).T)
            )                                                          # (n, no)

            Ca = A + B + C                                             # (n, no)

            if self.normalise:
                ma = ma / math.sqrt(ni)
                Ca = Ca / ni

            # ── Activation moments ───────────────────────────────────
            if activation == "sigmoid":
                t = torch.sqrt(1 + _PI / 8 * Ca)
                my = _sigmoid(ma / t)
                Cy = my * (1 - my) * (1 - 1 / t) + self.noise

            elif activation == "relu":
                E1 = ma
                E2 = ma ** 2 + Ca
                Cg = Ca * _gauss_pdf(torch.zeros_like(ma), ma, Ca)
                Phi = _STD_NORMAL.cdf(ma / torch.sqrt(Ca))

                my = E1 * Phi + Cg
                Cy = E2 * Phi + ma * Cg - my ** 2 + self.noise

            elif activation == "linear":
                my = ma
                Cy = Ca + self.noise

            else:
                raise ValueError(f"Unknown activation {activation!r}")

            all_ma.append(ma)
            all_Ca.append(Ca)
            all_my.append(my)
            all_Cy.append(Cy)

            # ── Prepare input for next layer ─────────────────────────
            mz = my
            if self.use_bias:
                mz_ = torch.cat([mz, self._ones(n_samples, 1)], dim=1)
            else:
                mz_ = mz

            if self.use_bias:
                Cz_ = torch.diag_embed(
                    torch.cat([Cy, self._zeros(n_samples, 1)], dim=1)
                )
            else:
                Cz_ = torch.diag_embed(Cy)

        if training:
            # Squeeze batch dim (training is single-sample).
            return (
                [m[0] for m in all_my],
                [c[0] for c in all_Cy],
                [m[0] for m in all_ma],
                [c[0] for c in all_Ca],
            )
        return all_my[-1], all_Cy[-1]

    # ── Backward pass (RTS smoothing) ─────────────────────────────────────

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor):
        """Online update via backward smoothing.  x is (input_dim,), y (output_dim,)."""
        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)
        my, Cy, ma, Ca = self.forward(x.unsqueeze(0), training=True)

        # Target is observed exactly (zero observation noise).
        my_new = y
        Cy_new = torch.zeros_like(y)

        for i in reversed(range(self.n_l - 1)):
            ni = self.layers[i] + (1 if self.use_bias else 0)
            no = self.layers[i + 1]
            activation = self.act_fns[i]

            # Previous layer's activation (input to this layer).
            if i == 0:
                mz = x
                Cz = torch.zeros_like(x)
            else:
                mz = my[i - 1]
                Cz = Cy[i - 1]

            if self.use_bias:
                mz_ = torch.cat([mz, self._ones(1)])
                Cz_ = torch.cat([Cz, self._zeros(1)])
            else:
                mz_ = mz
                Cz_ = Cz

            # ── Cov(y, a): activation-to-preactivation covariance ────
            if activation == "sigmoid":
                t = torch.sqrt(1 + _PI / 8 * Ca[i])
                Cya = (
                    math.sqrt(_PI / 8) * Ca[i] / t
                    * _gauss_pdf(
                        math.sqrt(_PI / 8) * ma[i] / t,
                        self._zeros(no),
                        self._ones(no),
                    )
                )

            elif activation == "relu":
                E2 = ma[i] ** 2 + Ca[i]
                Cg = Ca[i] * _gauss_pdf(torch.zeros_like(ma[i]), ma[i], Ca[i])
                Phi = _STD_NORMAL.cdf(ma[i] / torch.sqrt(Ca[i]))
                Cya = E2 * Phi + ma[i] * Cg - my[i] * ma[i]

            elif activation == "linear":
                Cya = ma[i] ** 2 + Ca[i] - my[i] * ma[i]

            else:
                raise ValueError(f"Unknown activation {activation!r}")

            # ── Scalar Kalman gain per neuron ────────────────────────
            k = Cya / Cy[i]                                           # (no,)
            da = k * (my_new - my[i])                                  # (no,)
            Da = (k ** 2) * (Cy_new - Cy[i])                          # (no,)

            # ── Cross-covariances for weight and input updates ───────
            if self.normalise:
                Cwa = self.Cw[i] @ mz_ / math.sqrt(ni)                # (no, ni)
                Cza = (Cz_.unsqueeze(-1).expand(ni, no)
                       * self.mw[i] / ni)                              # (ni, no)
            else:
                Cwa = self.Cw[i] @ mz_                                 # (no, ni)
                Cza = Cz_.unsqueeze(-1).expand(ni, no) * self.mw[i]    # (ni, no)

            Ca_inv = 1 / Ca[i]                                         # (no,)

            # Kalman gains for weights and inputs.
            L_up = Cwa * Ca_inv.unsqueeze(1)                           # (no, ni)
            L_low = Cza * Ca_inv.unsqueeze(0)                          # (ni, no)

            # ── Weight mean update ───────────────────────────────────
            self.mw[i] = self.mw[i] + (L_up * da.unsqueeze(1)).T      # (ni, no)

            # ── Propagate target backward ────────────────────────────
            my_new = mz + L_low[:-1] @ da                              # (layers[i],)

            # ── Weight covariance, rank-1 per neuron j: Da[j] * outer(L_up[j]) ──
            self.Cw[i] = self.Cw[i] + (
                Da.view(no, 1, 1) * L_up.unsqueeze(2) * L_up.unsqueeze(1)
            )

            # ── Input variance update ────────────────────────────────
            G = L_low ** 2 * Da.unsqueeze(0)                           # (ni, no)
            Cy_new = Cz + G[:-1].sum(dim=1)                            # (layers[i],)

    # ── Training loop ─────────────────────────────────────────────────────

    @torch.no_grad()
    def fit(self, X: Tensor, Y: Tensor, epochs: int = 1):
        """Online training: one forward + backward per sample."""
        X = X.to(device=self.device, dtype=self.dtype)
        Y = Y.to(device=self.device, dtype=self.dtype)
        losses = []
        for _epoch in range(epochs):
            for i in range(X.shape[0]):
                # Loss before the update.
                my_out, Cy_out = self.forward(X[i].unsqueeze(0), training=False)
                var = torch.clamp(Cy_out.squeeze(0), min=1e-8)
                nll = 0.5 * (
                    torch.log(2 * _PI * var)
                    + (Y[i] - my_out.squeeze(0)) ** 2 / var
                ).mean()
                losses.append(nll.item())
                self.update(X[i], Y[i])
        return losses

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Batch prediction returning (mean, variance)."""
        X = X.to(device=self.device, dtype=self.dtype)
        mean, var = self.forward(X, training=False)
        return mean, var


# ── BNNMethod adapter ────────────────────────────────────────────────────────

class WagnerKBNNMethod(BNNMethod):
    """Wagner et al. (2023) diagonal KBNN wrapped for the experiment runner."""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        """Translate the experiment config into a _WagnerBNN."""
        super().__init__(input_dim, output_dim, config)

        cfg = dict(config)
        hidden = cfg.pop("hidden_sizes", [50])
        activation = cfg.pop("activation", "relu")
        output_activation = cfg.pop("output_activation", "linear")
        use_bias = cfg.pop("use_bias", True)
        noise = cfg.pop("noise", 0.01)
        normalise = cfg.pop("normalise", False)
        self.epochs = cfg.pop("epochs", 1)
        weight_prior = cfg.pop("weight_prior", "scalar")
        weight_init_scale = cfg.pop("weight_init_scale", 1.0)
        weight_cov_init_scale = cfg.pop("weight_cov_init_scale", 1.0)
        dtype, device = pop_dtype_device(cfg)

        layer_sizes = [input_dim] + hidden + [output_dim]
        n_hidden = len(hidden)
        act_fns = [activation] * n_hidden + [output_activation]

        self._net = _WagnerBNN(
            layer_sizes, act_fns,
            use_bias=use_bias, noise=noise, normalise=normalise, dtype=dtype, device=device,
            weight_prior=weight_prior,
            weight_init_scale=weight_init_scale,
            weight_cov_init_scale=weight_cov_init_scale,
        )

    def _fit(self, X: Tensor, Y: Tensor) -> dict:
        """Stream the data through the smoother once per epoch."""
        losses = self._net.fit(X, Y, epochs=self.epochs)
        return {"losses": losses}

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and marginal variance."""
        return self._net.predict(X)

    @property
    def net(self) -> _WagnerBNN:
        """The underlying network, for code that needs its internals."""
        return self._net
