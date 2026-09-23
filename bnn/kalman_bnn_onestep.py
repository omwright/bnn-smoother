"""
The proposed method: Algorithm 2 of the paper.

A forward pass propagates moments through the network; a backward pass smooths
the observation back through it and updates the weight posterior in closed
form.  The network is read as activation → affine, so each layer takes a
*single* Kalman gain that updates its input x and its weights W jointly, where
Wagner's KBNN takes two gains per layer.  A 2-hidden-layer ReLU network with
linear output therefore needs 3 backward updates rather than 5.

A sigmoid output activation has no affine after it to absorb, so the backward
pass starts with one terminal smoothing step and then runs the layer loop.

The key identity is Theorem 1: for Gaussian x with z = g(x) element-wise,
Σ_xz = Σ_xx · diag(∂μ_z/∂μ_x), the derivative being Φ(μ/σ) for ReLU and
h(1−h)/t for the probit-approximated sigmoid.
"""

import math
from typing import List, Tuple

import torch
from torch import Tensor

from .covariance import CholeskyCovariance, DenseCovariance
from .device import cpu_randn
from .moments import (
    relu_covariance,
    relu_moment,
    sigmoid_covariance,
    sigmoid_moment,
    standard_normal_cdf,
)

# Probit approximation constant, as in moments.py.
_LAMBDA_SQ = math.pi / 8.0


DEFAULTS = {
    "layer_sizes": [1, 50, 1],
    "activation": "relu",
    "output_activation": "linear",
    "use_bias": True,
    "cov_type": "dense",  # "cholesky" or "dense"
    "diagonal_loading": 0.0,
    # ── Noise ────────────────────────────────────────────────────────────
    "process_noise": None,       # float or "prior_scaled"; None → fall back
    "process_noise_scale": 0.01,
    "likelihood_noise": None,    # float (σ²); None → fall back
    # Legacy (backward compatible):
    "observation_noise": 1e-4,
    # ────────────────────────────────────────────────────────────────────
    "weight_prior": "scalar",    # "scalar" or "he"
    "weight_init_scale": 1.0,
    "weight_cov_init_scale": 1.0,
    "relu_cov_order": 4,
    "sigmoid_cov_order": 4,
    "min_eigval": 0.0,
    "dtype": "float64",
    "device": "cpu",
}


def _get_dtype(name: str) -> torch.dtype:
    """Map a config dtype name to a torch dtype."""
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _make_covariance(dim, cov_type, init_scale, loading, device, dtype):
    """Build one layer's weight-covariance store."""
    cls = CholeskyCovariance if cov_type == "cholesky" else DenseCovariance
    return cls(dim=dim, init_scale=init_scale, loading=loading, device=device, dtype=dtype)


class Layer:
    """Single dense layer with weight mean and covariance."""

    def __init__(self, n_in: int, n_out: int, weight_mean: Tensor, weight_cov):
        """Hold a (n_in, n_out) weight mean and its covariance store."""
        self.n_in = n_in
        self.n_out = n_out
        self.weight_mean = weight_mean
        self._cov = weight_cov

    @property
    def weight_cov(self) -> Tensor:
        """4D view (n_in, n_out, n_in, n_out) for einsum. Free reshape."""
        return self._cov.get_covariance().view(self.n_in, self.n_out, self.n_in, self.n_out)

    def set_weight_cov(self, cov_4d: Tensor):
        """Set from 4D tensor."""
        self._cov.set_covariance(cov_4d.reshape(self.n_in * self.n_out, -1))


class KBNNOneStep:
    """
    BNN with moment propagation (forward) and one-step RTS smoothing (update).

    Weight means and covariances are stored per layer; see ``Layer`` above
    and ``bnn/covariance.py`` for the storage layout.
    """

    def __init__(self, config: dict = None):
        """Build the layers and draw weights from the configured prior."""
        cfg = {**DEFAULTS, **(config or {})}
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        self.dtype = _get_dtype(cfg["dtype"])

        sizes = cfg["layer_sizes"]
        self.n_layers = len(sizes) - 1
        self.layers = []

        prior_mode = cfg["weight_prior"]
        if prior_mode not in ("scalar", "he"):
            raise ValueError(
                f"Unknown weight_prior {prior_mode!r}. "
                f"Choose 'scalar' (flat, backward compatible) or 'he' (per-layer He/NNGP)."
            )

        # ── Build layers, collecting per-layer prior variances ───────
        self._prior_vars: List[float] = []

        for i in range(self.n_layers):
            n_in = sizes[i] + (1 if cfg["use_bias"] else 0)
            n_out = sizes[i + 1]

            mean_std, cov_scale = self._prior_scales(prior_mode, n_in, cfg)
            self._prior_vars.append(cov_scale)

            W = cpu_randn(n_in, n_out, device=self.device, dtype=self.dtype)
            W *= mean_std
            if cfg["use_bias"]:
                W[-1, :] = 0

            cov = _make_covariance(
                n_in * n_out, cfg["cov_type"], cov_scale,
                cfg["diagonal_loading"], self.device, self.dtype
            )
            self.layers.append(Layer(n_in, n_out, W, cov))

        self._process_noise_per_layer = self._resolve_process_noise(cfg)
        self._likelihood_noise = self._resolve_likelihood_noise(cfg)

    # -----------------------------------------------------------------
    # Noise / prior resolution
    # -----------------------------------------------------------------

    def _resolve_process_noise(self, cfg: dict) -> List[float]:
        """Per-layer process noise: a constant, or scaled by the prior."""
        pn = cfg["process_noise"]
        if pn is None:
            val = float(cfg["observation_noise"])
            return [val] * self.n_layers
        if isinstance(pn, str) and pn == "prior_scaled":
            scale = float(cfg["process_noise_scale"])
            return [scale * pv for pv in self._prior_vars]
        val = float(pn)
        return [val] * self.n_layers

    @staticmethod
    def _resolve_likelihood_noise(cfg: dict) -> float:
        """Aleatoric noise variance σ², or 0 when unset."""
        ln = cfg["likelihood_noise"]
        if ln is None:
            return 0.0
        return float(ln)

    @staticmethod
    def _prior_scales(mode: str, n_in: int, cfg: dict) -> Tuple[float, float]:
        """Return (mean std, covariance scale) for one layer's prior."""
        if mode == "he":
            prior_var = cfg["weight_cov_init_scale"] * 2.0 / n_in
            return math.sqrt(prior_var), prior_var
        else:
            return cfg["weight_init_scale"] / math.sqrt(n_in), cfg["weight_cov_init_scale"]

    # -----------------------------------------------------------------
    # Layer convention helpers
    # -----------------------------------------------------------------

    def _preceding_activation(self, i: int) -> str:
        """Activation applied to the input of affine_i.

        Under the activation→affine convention this is the identity for layer
        0, and ``cfg["activation"]`` thereafter.  ``cfg["output_activation"]``
        sits after the final affine, outside the layer loop, and ``update``
        handles it as a terminal smoothing pass.
        """
        if i == 0:
            return "linear"
        return self.cfg["activation"]

    def _eye(self, n: int) -> Tensor:
        """Identity on this net's device and dtype."""
        return torch.eye(n, device=self.device, dtype=self.dtype)

    def _zeros(self, *shape) -> Tensor:
        """Zeros on this net's device and dtype."""
        return torch.zeros(*shape, device=self.device, dtype=self.dtype)

    def _cast(self, t: Tensor) -> Tensor:
        """Move an incoming tensor onto this net's device and dtype."""
        return t.to(device=self.device, dtype=self.dtype)

    # -----------------------------------------------------------------
    # Activation moments with cross-covariance (Theorem 1)
    # -----------------------------------------------------------------

    def _activation_moments(self, act_name: str, mu: Tensor, cov: Tensor):
        """Return (z_mean, z_cov, deriv) for z = g(x).

        ``deriv`` is the diagonal of ∂μ_z/∂μ_x; the full cross-covariance
        Σ_xz follows from Theorem 1 as ``cov * deriv.unsqueeze(0)``.

        For the identity activation the derivative is the all-ones vector,
        so Σ_xz = Σ_xx, Σ_zz = Σ_xx, and μ_z = μ_x (pass-through).
        """
        if act_name == "linear":
            deriv = torch.ones(mu.shape[0], device=self.device, dtype=self.dtype)
            return mu, cov, deriv

        if act_name == "relu":
            z_mean, z_cov = self._forward_relu(mu, cov)
            sigma = torch.sqrt(torch.clamp(cov.diagonal(), min=1e-10))
            deriv = standard_normal_cdf(mu / sigma)
            return z_mean, z_cov, deriv

        if act_name == "sigmoid":
            z_mean, z_cov = self._forward_sigmoid(mu, cov)
            sigma_sq = torch.clamp(cov.diagonal(), min=1e-10)
            t = torch.sqrt(1.0 + _LAMBDA_SQ * sigma_sq)
            h = torch.sigmoid(mu / t)
            deriv = h * (1.0 - h) / t
            return z_mean, z_cov, deriv

        raise ValueError(
            f"Unknown activation {act_name!r}. Supported: 'relu', 'sigmoid', 'linear'."
        )

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, x: Tensor, return_state: bool = False):
        """Propagate moments through the network.

        Returns (mean, cov), or (mean, cov, states) when ``return_state``,
        where states[i] holds what layer i's backward Kalman gain needs.  The
        moments are those of the latent output after the optional output
        activation and EXCLUDE ``likelihood_noise``, which ``predict`` and
        ``_nll`` add.
        """
        x = self._cast(x)

        # Previous affine's moments; for layer 0, the input with zero covariance.
        y_mean = x
        y_cov = self._zeros(x.shape[0], x.shape[0])

        states = [] if return_state else None

        for i, layer in enumerate(self.layers):
            # ── 1. Preceding activation: z = g(x), x = previous y ──────
            x_mean, x_cov = y_mean, y_cov
            act_name = self._preceding_activation(i)
            z_nobias_mean, z_nobias_cov, deriv = self._activation_moments(
                act_name, x_mean, x_cov
            )

            # Theorem 1: Σ_xz = Σ_xx · diag(deriv).  Backward pass builds Σ_xy from it.
            sigma_xz_nobias = x_cov * deriv.unsqueeze(0)

            # ── 2. Append bias to z ────────────────────────────────────
            if self.cfg["use_bias"]:
                n = z_nobias_mean.shape[0]
                z_mean = torch.cat(
                    [z_nobias_mean, torch.ones(1, device=self.device, dtype=self.dtype)]
                )
                z_cov = self._zeros(n + 1, n + 1)
                z_cov[:n, :n] = z_nobias_cov
            else:
                z_mean, z_cov = z_nobias_mean, z_nobias_cov

            # ── 3. Affine: y = W^T z ───────────────────────────────────
            y_mean, y_cov = self._forward_affine(z_mean, z_cov, layer)

            # ── 4. Σ_xy = Σ_xz · W̄ (paper eq. 20) ─────────────────────
            # Only the non-bias rows of W enter: the bias row is deterministic.
            n_x = x_mean.shape[0]
            W_no_bias = layer.weight_mean[:n_x, :]  # (n_x, n_out)
            sigma_xy = sigma_xz_nobias @ W_no_bias  # (n_x, n_out)

            # ── 5. Process noise ───────────────────────────────────────
            pn = self._process_noise_per_layer[i]
            y_cov = y_cov + pn * self._eye(y_cov.shape[0])

            if return_state:
                states.append({
                    "x_mean": x_mean, "x_cov": x_cov,
                    "z_mean": z_mean, "z_cov": z_cov,
                    "y_mean": y_mean, "y_cov": y_cov,
                    "sigma_xy": sigma_xy,
                    "act_name": act_name,
                })

        # ── 6. Optional terminal output activation ─────────────────────
        out_act = self.cfg["output_activation"]
        if out_act == "linear":
            final_mean, final_cov = y_mean, y_cov
        elif out_act == "relu":
            final_mean, final_cov = self._forward_relu(y_mean, y_cov)
        elif out_act == "sigmoid":
            final_mean, final_cov = self._forward_sigmoid(y_mean, y_cov)
        else:
            raise ValueError(
                f"Unknown output_activation {out_act!r}. "
                f"Supported: 'relu', 'sigmoid', 'linear'."
            )

        return (final_mean, final_cov, states) if return_state else (final_mean, final_cov)

    def _forward_affine(self, z_mean: Tensor, z_cov: Tensor, layer: Layer):
        """Affine moments, y = Wᵀz."""
        W, C = layer.weight_mean, layer.weight_cov

        y_mean = z_mean @ W
        A = W.T @ z_cov @ W                                   # input uncertainty
        B = torch.einsum('iojp,i,j->op', C, z_mean, z_mean)   # weight uncertainty
        D = torch.einsum('iojp,ij->op', C, z_cov)             # interaction

        return y_mean, A + B + D

    def _forward_relu(self, mu: Tensor, cov: Tensor):
        """ReLU moments via the Hadamard series."""
        out_cov = relu_covariance(
            mu.unsqueeze(0), cov.unsqueeze(0),
            order=self.cfg["relu_cov_order"]
        ).squeeze(0)
        sigma = torch.sqrt(torch.clamp(cov.diagonal(), min=1e-10))
        out_mean = relu_moment(mu, sigma)
        return out_mean, out_cov

    def _forward_sigmoid(self, mu: Tensor, cov: Tensor):
        """Sigmoid moments via probit and the Hadamard series."""
        out_cov = sigmoid_covariance(
            mu.unsqueeze(0), cov.unsqueeze(0),
            order=self.cfg["sigmoid_cov_order"]
        ).squeeze(0)
        sigma = torch.sqrt(torch.clamp(cov.diagonal(), min=1e-10))
        out_mean = sigmoid_moment(mu, sigma)
        return out_mean, out_cov

    # -------------------------------------------------------------------------
    # Update (one-step RTS smoothing)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor):
        """Update weight posteriors via one-step-per-layer RTS smoothing."""
        x = self._cast(x)
        y = self._cast(y)
        _, _, states = self.forward(x, return_state=True)

        loading = self.cfg["diagonal_loading"]

        # Backward target, on the last affine's output space.
        tgt_mean, tgt_cov = self._initialize_target(states[-1], y)

        # ── One-step backward loop ────────────────────────────────────
        for i in reversed(range(self.n_layers)):
            s = states[i]
            layer = self.layers[i]
            # The returned target lives on x^(i) = y^(i-1), the previous
            # layer's affine output, or the network input when i = 0.
            tgt_mean, tgt_cov = self._smooth_layer(
                tgt_mean, tgt_cov, s, layer, loading
            )

    def _initialize_target(self, last_state: dict, y: Tensor):
        """Build the smoother's initial target on the last affine's output.

        The observation lives after the output activation, so: take the
        forward moments there, Kalman-update them with the observation, and
        smooth back through the activation to the space the layer loop wants.
        """
        y_mean_last = last_state["y_mean"]   # pre output activation
        y_cov_last = last_state["y_cov"]
        n_out = y.shape[0]
        loading = self.cfg["diagonal_loading"]
        out_act = self.cfg["output_activation"]

        # (i) Forward moments on post-activation space.
        if out_act == "linear":
            post_mean, post_cov = y_mean_last, y_cov_last
        elif out_act == "relu":
            post_mean, post_cov = self._forward_relu(y_mean_last, y_cov_last)
        elif out_act == "sigmoid":
            post_mean, post_cov = self._forward_sigmoid(y_mean_last, y_cov_last)
        else:
            raise ValueError(f"Unknown output_activation {out_act!r}.")

        # (ii) Kalman update with the observation, R = likelihood_noise · I.
        if self._likelihood_noise > 0:
            R = self._likelihood_noise * self._eye(n_out)
            S = post_cov + R
            K = torch.linalg.solve(S, post_cov).T
            obs_mean = post_mean + K @ (y - post_mean)
            obs_cov = self._ensure_psd(post_cov - K @ post_cov)
        else:
            obs_mean = y
            obs_cov = self._zeros(n_out, n_out)

        # (iii) Smooth back through a non-trivial output activation.
        if out_act == "linear":
            return obs_mean, obs_cov
        elif out_act == "relu":
            return self._smooth_terminal_relu(
                obs_mean, obs_cov, y_mean_last, y_cov_last,
                post_mean, post_cov, loading
            )
        elif out_act == "sigmoid":
            return self._smooth_terminal_sigmoid(
                obs_mean, obs_cov, y_mean_last, y_cov_last,
                post_mean, post_cov, loading
            )

    def _smooth_layer(self, tgt_mean, tgt_cov, state, layer, loading):
        """One-step RTS update for a single layer, eqs. (31)-(35).

        From a target on the layer's affine output y, computes one Kalman gain
        K = [K_x; K_w] and updates the layer input x and the weights W jointly.
        Assuming x and W independent lets K be built from the two
        cross-covariances Σ_xy and Σ_wy separately.
        """
        n_in, n_out = layer.n_in, layer.n_out
        W_cov = layer.weight_cov

        # Σ_xy came from the forward pass, via Theorem 1 and eq. 20.
        sigma_xy = state["sigma_xy"]                              # (n_x, n_out)

        # Σ_wy = Σ_ww · (μ_z ⊗ I): the weight covariance contracted with μ_z.
        z_mean = state["z_mean"]                                  # (n_in,) w/ bias
        sigma_wy = torch.einsum('iojp,j->iop', W_cov, z_mean)     # (n_in, n_out, n_out)

        # One LU factorisation of Σ_yy serves both K_x and K_w -- part of what
        # makes the one-step formulation cheaper than the two-step smoother.
        y_cov = state["y_cov"]
        reg = y_cov + loading * self._eye(n_out)
        LU, piv = torch.linalg.lu_factor(reg)

        K_x = torch.linalg.lu_solve(LU, piv, sigma_xy.T).T        # (n_x, n_out)
        K_w = torch.linalg.lu_solve(
            LU, piv, sigma_wy.reshape(-1, n_out).T
        ).T.reshape(n_in, n_out, n_out)                           # (n_in, n_out, n_out)

        # ── Residuals (target − forward prediction) ──
        y_mean = state["y_mean"]
        diff_mean = tgt_mean - y_mean
        diff_cov = tgt_cov - y_cov

        # ── Weight update ──
        layer.weight_mean = layer.weight_mean + torch.einsum('iom,m->io', K_w, diff_mean)
        W_cov_new = W_cov + torch.einsum('iom,mn,jpn->iojp', K_w, diff_cov, K_w)
        layer.set_weight_cov(W_cov_new)

        # ── Input update — propagates as next iteration's target ──
        x_mean_new = state["x_mean"] + K_x @ diff_mean
        x_cov_new = self._ensure_psd(state["x_cov"] + K_x @ diff_cov @ K_x.T)

        return x_mean_new, x_cov_new

    # ------------------------------------------------------------------
    # Terminal activation smoothers (output activation only)
    # ------------------------------------------------------------------

    def _smooth_terminal_relu(self, tgt_mean, tgt_cov, pre_mean, pre_cov,
                              post_mean, post_cov, loading):
        """Smooth a target back through a terminal ReLU."""
        sigma = torch.sqrt(torch.clamp(pre_cov.diagonal(), min=1e-10))
        phi_z = standard_normal_cdf(pre_mean / sigma)
        cross = pre_cov * phi_z.unsqueeze(0)   # Σ_{pre, post} by Theorem 1

        reg = post_cov + loading * self._eye(post_cov.shape[0])
        K = torch.linalg.solve(reg, cross.T).T

        new_mean = pre_mean + K @ (tgt_mean - post_mean)
        new_cov = pre_cov + K @ (tgt_cov - post_cov) @ K.T
        return new_mean, self._ensure_psd(new_cov)

    def _smooth_terminal_sigmoid(self, tgt_mean, tgt_cov, pre_mean, pre_cov,
                                 post_mean, post_cov, loading):
        """Smooth a target back through a terminal sigmoid."""
        sigma_sq = torch.clamp(pre_cov.diagonal(), min=1e-10)
        t = torch.sqrt(1.0 + _LAMBDA_SQ * sigma_sq)
        h = torch.sigmoid(pre_mean / t)
        f1 = h * (1.0 - h) / t
        cross = pre_cov * f1.unsqueeze(0)

        reg = post_cov + loading * self._eye(post_cov.shape[0])
        K = torch.linalg.solve(reg, cross.T).T

        new_mean = pre_mean + K @ (tgt_mean - post_mean)
        new_cov = pre_cov + K @ (tgt_cov - post_cov) @ K.T
        return new_mean, self._ensure_psd(new_cov)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _ensure_psd(self, cov: Tensor) -> Tensor:
        """Clamp eigenvalues when ``min_eigval`` is set, else just symmetrise."""
        if self.cfg["min_eigval"] > 0.0:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = torch.clamp(eigvals, min=self.cfg["min_eigval"])
            return eigvecs @ torch.diag(eigvals) @ eigvecs.T
        else:
            return (cov + cov.T) * 0.5  # cheap symmetry enforcement

    # -------------------------------------------------------------------------
    # Training / Inference
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, X: Tensor, Y: Tensor, epochs: int = 1, verbose: bool = False) -> List[float]:
        """Online training.  Returns per-sample NLL."""
        X = self._cast(X)
        Y = self._cast(Y)
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(X.shape[0]):
                mean, cov = self.forward(X[i])
                nll = self._nll(Y[i], mean, cov)
                losses.append(nll.item())
                epoch_loss += nll.item()
                self.update(X[i], Y[i])
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, NLL: {epoch_loss/X.shape[0]:.4f}")
        return losses

    def _nll(self, y, mean, cov):
        """Negative log-likelihood under p(y|f) = N(f, σ²I)."""
        y = self._cast(y)
        diff = y - mean
        reg = cov + (
            self._likelihood_noise + self.cfg["diagonal_loading"]
        ) * self._eye(cov.shape[0])
        _, logdet = torch.linalg.slogdet(reg)
        quad = diff @ torch.linalg.solve(reg, diff)
        return 0.5 * (logdet + quad + mean.shape[0] * 1.8378770664093453)

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (means, variances) including aleatoric noise."""
        X = self._cast(X)
        single = X.dim() == 1
        if single:
            X = X.unsqueeze(0)

        means, vars = [], []
        for i in range(X.shape[0]):
            m, c = self.forward(X[i])
            means.append(m)
            vars.append(c.diagonal() + self._likelihood_noise)

        means, vars = torch.stack(means), torch.stack(vars)
        return (means.squeeze(0), vars.squeeze(0)) if single else (means, vars)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    net = KBNNOneStep({
        "layer_sizes": [1, 20, 1],
        "activation": "relu",
        "cov_type": "cholesky",
    })

    X = torch.linspace(-3, 3, 50, dtype=torch.float64).unsqueeze(1)
    Y = torch.sin(X) + 0.1 * torch.randn_like(X)

    net.fit(X, Y, epochs=1, verbose=True)

    means, vars = net.predict(X)
    print(f"Predictions: [{means.min():.3f}, {means.max():.3f}], "
          f"var: [{vars.min():.2e}, {vars.max():.2e}]")
