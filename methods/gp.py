"""
Exact GP baseline — online Gaussian process with GPyTorch.

The gold-standard probabilistic baseline on low-dimensional tasks: an exact
posterior, and so Bayes-optimal for its kernel class.  It does not scale to
high-dimensional input, which is why CartPole has it and the Industrial
Benchmark does not.

Each arriving pair is appended to the training set and the posterior rebuilt.
Exact GPs cost O(n³), so ``max_train_size`` caps the set with a sliding window.

Two output modes:

``independent``  one GP per output dimension, no cross-output covariance, so
                 ``predict`` gives marginal variances and ``forward`` is absent.
``multitask``    an Intrinsic Coregionalization Model, learning a task
                 covariance B so that K_output = B ⊗ K_input.  ``forward``
                 then returns the full covariance, enabling ``nll_full``.

Config keys
-----------
mode            str     "independent" or "multitask"      "independent"
max_train_size  int     sliding window cap (0 = unlimited) 500
kernel          str     "rbf" or "matern"                 "rbf"
task_rank       int     rank of ICM task covariance        0 (= full rank)
learn_hyperparams  bool  optimize kernel hyperparams       false
optim_steps     int     hyperparameter optimization steps  10
lr              float   hyperparameter learning rate       0.01
dtype           str     "float32" or "float64"            "float64"
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import BNNMethod, pop_dtype_device

try:
    import gpytorch
    from gpytorch.distributions import (
        MultitaskMultivariateNormal,
        MultivariateNormal,
    )
    from gpytorch.kernels import (
        MaternKernel,
        MultitaskKernel,
        RBFKernel,
        ScaleKernel,
    )
    from gpytorch.likelihoods import (
        GaussianLikelihood,
        MultitaskGaussianLikelihood,
    )
    from gpytorch.means import ConstantMean, MultitaskMean
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.models import ExactGP
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


def _check_gpytorch():
    """Raise a helpful ImportError when GPyTorch is missing."""
    if not HAS_GPYTORCH:
        raise ImportError(
            "gpytorch is required for the GP baseline.  Install it with:\n"
            "  uv sync --extra baselines"
        )


def _make_base_kernel(kernel_name: str):
    """RBF or Matérn-5/2 kernel."""
    if kernel_name == "matern":
        return MaternKernel(nu=2.5)
    return RBFKernel()


# =============================================================================
# Independent GP (one per output dimension)
# =============================================================================

if HAS_GPYTORCH:

    class _SingleOutputGPModel(ExactGP):
        """Single-output exact GP with an RBF or Matérn kernel."""

        def __init__(self, train_x, train_y, likelihood, kernel_name="rbf"):
            """Constant mean, scaled RBF or Matérn kernel."""
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(_make_base_kernel(kernel_name))

        def forward(self, x):
            """Prior at ``x``."""
            return MultivariateNormal(
                self.mean_module(x), self.covar_module(x),
            )


class _OnlineIndependentGP:
    """One independent GP per output dimension, with no cross-output covariance."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_train_size: int = 500,
        kernel: str = "rbf",
        learn_hyperparams: bool = False,
        optim_steps: int = 10,
        lr: float = 0.01,
        dtype: torch.dtype = torch.float64,
        device: torch.device = None,
    ):
        """Set up an empty training set; models are built on first update."""
        _check_gpytorch()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_train_size = max_train_size
        self.kernel_name = kernel
        self.learn_hyperparams = learn_hyperparams
        self.optim_steps = optim_steps
        self.lr = lr
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        self._X_train: list[Tensor] = []
        self._Y_train: list[Tensor] = []

        self._models: list | None = None
        self._likelihoods: list | None = None

    # ── Internal ─────────────────────────────────────────────────────

    def _rebuild_models(self):
        """Refit one GP per output dimension on the current training set."""
        X = torch.stack(self._X_train).to(device=self.device, dtype=self.dtype)
        Y = torch.stack(self._Y_train).to(device=self.device, dtype=self.dtype)

        self._models = []
        self._likelihoods = []

        for j in range(self.output_dim):
            likelihood = GaussianLikelihood().to(device=self.device, dtype=self.dtype)
            model = _SingleOutputGPModel(
                X, Y[:, j], likelihood, self.kernel_name,
            ).to(device=self.device, dtype=self.dtype)

            if self.learn_hyperparams and X.shape[0] > 5:
                model.train()
                likelihood.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                mll = ExactMarginalLogLikelihood(likelihood, model)
                for _ in range(self.optim_steps):
                    optimizer.zero_grad()
                    loss = -mll(model(X), Y[:, j])
                    loss.backward()
                    optimizer.step()

            model.eval()
            likelihood.eval()
            self._models.append(model)
            self._likelihoods.append(likelihood)

    def _apply_window(self):
        """Drop the oldest points beyond ``max_train_size``."""
        if self.max_train_size > 0 and len(self._X_train) > self.max_train_size:
            self._X_train = self._X_train[-self.max_train_size:]
            self._Y_train = self._Y_train[-self.max_train_size:]

    # ── Public interface ─────────────────────────────────────────────

    @torch.enable_grad()
    def update(self, x: Tensor, y: Tensor):
        """Append one pair and refit."""
        self._X_train.append(x.detach().clone().to(device=self.device, dtype=self.dtype))
        self._Y_train.append(y.detach().clone().to(device=self.device, dtype=self.dtype))
        self._apply_window()
        self._rebuild_models()

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Marginal predictions, both (n, output_dim).  Wide prior if untrained."""
        if self._models is None or len(self._X_train) == 0:
            n = X.shape[0]
            return (torch.zeros(n, self.output_dim, dtype=self.dtype, device=self.device),
                    torch.ones(n, self.output_dim, dtype=self.dtype, device=self.device) * 1e2)

        X = X.to(device=self.device, dtype=self.dtype)
        means, vars_ = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for j in range(self.output_dim):
                pred = self._likelihoods[j](self._models[j](X))
                means.append(pred.mean)
                vars_.append(pred.variance)

        return torch.stack(means, dim=-1), torch.stack(vars_, dim=-1)

    def fit(self, X: Tensor, Y: Tensor, epochs: int = 1):
        """Add every pair, then refit once."""
        for i in range(X.shape[0]):
            self._X_train.append(X[i].detach().clone().to(device=self.device, dtype=self.dtype))
            self._Y_train.append(Y[i].detach().clone().to(device=self.device, dtype=self.dtype))
        self._apply_window()
        self._rebuild_models()
        return []


# =============================================================================
# Multitask GP (Intrinsic Coregionalization Model)
# =============================================================================

if HAS_GPYTORCH:

    class _MultitaskGPModel(ExactGP):
        """Multitask exact GP with an ICM kernel.

        The ICM factorises the (n·T) × (n·T) covariance as B ⊗ K_input, with B
        the T×T task covariance.  ``task_rank`` sets the rank of B; 0 is full.
        """

        def __init__(self, train_x, train_y, likelihood,
                     num_tasks: int, kernel_name: str = "rbf",
                     task_rank: int = 0):
            """Multitask constant mean and ICM kernel over ``num_tasks``."""
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
            rank = task_rank if task_rank > 0 else num_tasks
            self.covar_module = MultitaskKernel(
                ScaleKernel(_make_base_kernel(kernel_name)),
                num_tasks=num_tasks,
                rank=rank,
            )

        def forward(self, x):
            """Prior at ``x``."""
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return MultitaskMultivariateNormal(mean, covar)


class _OnlineMultitaskGP:
    """Multitask GP with an ICM kernel, online.

    Learns cross-output correlations through the task covariance B, and
    exposes ``forward_full`` so the runner can compute the full NLL.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_train_size: int = 500,
        kernel: str = "rbf",
        task_rank: int = 0,
        learn_hyperparams: bool = False,
        optim_steps: int = 10,
        lr: float = 0.01,
        dtype: torch.dtype = torch.float64,
        device: torch.device = None,
    ):
        """Set up an empty training set; the model is built on first update."""
        _check_gpytorch()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_train_size = max_train_size
        self.kernel_name = kernel
        self.task_rank = task_rank
        self.learn_hyperparams = learn_hyperparams
        self.optim_steps = optim_steps
        self.lr = lr
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        self._X_train: list[Tensor] = []
        self._Y_train: list[Tensor] = []

        self._model: _MultitaskGPModel | None = None
        self._likelihood: MultitaskGaussianLikelihood | None = None

    # ── Internal ─────────────────────────────────────────────────────

    def _rebuild_model(self):
        """Refit the multitask GP on the current training set."""
        X = torch.stack(self._X_train).to(device=self.device, dtype=self.dtype)  # (n, input_dim)
        Y = torch.stack(self._Y_train).to(device=self.device, dtype=self.dtype)  # (n, output_dim)

        likelihood = MultitaskGaussianLikelihood(
            num_tasks=self.output_dim,
        ).to(device=self.device, dtype=self.dtype)
        model = _MultitaskGPModel(
            X, Y, likelihood,
            num_tasks=self.output_dim,
            kernel_name=self.kernel_name,
            task_rank=self.task_rank,
        ).to(device=self.device, dtype=self.dtype)

        if self.learn_hyperparams and X.shape[0] > 5:
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(self.optim_steps):
                optimizer.zero_grad()
                loss = -mll(model(X), Y)
                loss.backward()
                optimizer.step()

        model.eval()
        likelihood.eval()
        self._model = model
        self._likelihood = likelihood

    def _apply_window(self):
        """Drop the oldest points beyond ``max_train_size``."""
        if self.max_train_size > 0 and len(self._X_train) > self.max_train_size:
            self._X_train = self._X_train[-self.max_train_size:]
            self._Y_train = self._Y_train[-self.max_train_size:]

    # ── Public interface ─────────────────────────────────────────────

    @torch.enable_grad()
    def update(self, x: Tensor, y: Tensor):
        """Append one pair and refit."""
        self._X_train.append(x.detach().clone().to(device=self.device, dtype=self.dtype))
        self._Y_train.append(y.detach().clone().to(device=self.device, dtype=self.dtype))
        self._apply_window()
        self._rebuild_model()

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Marginal predictions → (mean, variance), shape (n, output_dim)."""
        if self._model is None or len(self._X_train) == 0:
            n = X.shape[0]
            return (torch.zeros(n, self.output_dim, dtype=self.dtype, device=self.device),
                    torch.ones(n, self.output_dim, dtype=self.dtype, device=self.device) * 1e2)

        X = X.to(device=self.device, dtype=self.dtype)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(X))
            return pred.mean, pred.variance

    def forward_full(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Full predictive covariance for one input.

        x is (input_dim,); returns (output_dim,) and (output_dim, output_dim).
        """
        if self._model is None or len(self._X_train) == 0:
            d = self.output_dim
            return (torch.zeros(d, dtype=self.dtype, device=self.device),
                    torch.eye(d, dtype=self.dtype, device=self.device) * 1e2)

        x = x.to(device=self.device, dtype=self.dtype).unsqueeze(0)  # (1, input_dim)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(x))
            mean = pred.mean.squeeze(0)          # (output_dim,)
            full_cov = pred.covariance_matrix    # (T, T) for a single point
            d = self.output_dim
            if full_cov.shape != (d, d):
                full_cov = full_cov[:d, :d]
            return mean, full_cov

    def fit(self, X: Tensor, Y: Tensor, epochs: int = 1):
        """Add every pair, then refit once."""
        for i in range(X.shape[0]):
            self._X_train.append(X[i].detach().clone().to(device=self.device, dtype=self.dtype))
            self._Y_train.append(Y[i].detach().clone().to(device=self.device, dtype=self.dtype))
        self._apply_window()
        self._rebuild_model()
        return []


# =============================================================================
# BNNMethod adapter
# =============================================================================

class GPMethod(BNNMethod):
    """Adapter: the online GPs <-> ``BNNMethod``.

    In multitask mode it also exposes ``forward``, for the full NLL.
    """

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        """Translate the experiment config into an online GP."""
        super().__init__(input_dim, output_dim, config)

        cfg = dict(config)
        mode = cfg.pop("mode", "independent")
        max_train_size = cfg.pop("max_train_size", 500)
        kernel = cfg.pop("kernel", "rbf")
        task_rank = cfg.pop("task_rank", 0)
        learn_hyperparams = cfg.pop("learn_hyperparams", False)
        optim_steps = cfg.pop("optim_steps", 10)
        lr = cfg.pop("lr", 0.01)
        dtype, device = pop_dtype_device(cfg)

        common = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            max_train_size=max_train_size,
            kernel=kernel,
            learn_hyperparams=learn_hyperparams,
            optim_steps=optim_steps,
            lr=lr,
            dtype=dtype,
            device=device,
        )

        self._mode = mode

        if mode == "multitask":
            self._net = _OnlineMultitaskGP(task_rank=task_rank, **common)
        else:
            self._net = _OnlineIndependentGP(**common)

    # ── BNNMethod interface ──────────────────────────────────────────

    def _fit(self, X: Tensor, Y: Tensor) -> dict:
        """Add every pair to the training set and refit."""
        self._net.fit(X, Y)
        return {}

    def predict(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and marginal variance."""
        return self._net.predict(X)

    def forward(self, x: Tensor, **kw):
        """Full predictive covariance for one input, multitask mode only.

        Raises AttributeError in independent mode, so the runners'
        ``hasattr(method, "forward")`` check falls through.
        """
        if self._mode != "multitask":
            raise AttributeError(
                "forward() requires mode='multitask' for full output covariance"
            )
        return self._net.forward_full(x)

    @property
    def net(self):
        """The underlying GP, for the runners' ``update_single``."""
        return self._net
