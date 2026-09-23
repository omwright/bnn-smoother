import warnings
from abc import ABC, abstractmethod

import torch
from torch import Tensor

# Weight covariance layout, for a weight matrix W of shape (n_in, n_out):
#
#   2D: (n_in * n_out, n_in * n_out)     as stored
#   4D: (n_in, n_out, n_in, n_out)       cov_4d[i1, o1, i2, o2] = Cov(W[i1, o1], W[i2, o2])
#
# The two are related by .view(), so converting between them is free.  The
# legacy (n_out, n_out, n_in, n_in) layout below needs a permute instead.


def cov_4d_to_2d(cov_4d: Tensor) -> Tensor:
    """Legacy 4D (out, out, in, in) to 2D (out*in, out*in).  Prefer .view()."""
    warnings.warn(
        "cov_4d_to_2d uses legacy (out,out,in,in) layout requiring permute. "
        "Consider using natural (in,out,in,out) layout with reshape instead.",
        DeprecationWarning,
        stacklevel=2
    )
    out1, out2, in1, in2 = cov_4d.shape
    cov_permuted = cov_4d.permute(0, 2, 1, 3)
    cov_2d = cov_permuted.reshape(out1 * in1, out2 * in2)
    return cov_2d


def cov_2d_to_4d(cov_2d: Tensor, out_dim: int, in_dim: int) -> Tensor:
    """Legacy 2D (out*in, out*in) to 4D (out, out, in, in).  Prefer .view()."""
    warnings.warn(
        "cov_2d_to_4d uses legacy (out,out,in,in) layout requiring permute. "
        "Consider using natural (in,out,in,out) layout with reshape instead.",
        DeprecationWarning,
        stacklevel=2
    )
    cov_4d_temp = cov_2d.reshape(out_dim, in_dim, out_dim, in_dim)
    cov_4d = cov_4d_temp.permute(0, 2, 1, 3)
    return cov_4d


class Covariance(ABC):
    """Abstract base for covariance representations."""

    def __init__(
        self,
        cov: Tensor = None,
        dim: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float64
    ):
        """Set up dimension, device and dtype from ``cov`` or ``dim``."""
        if cov is not None:
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                raise ValueError("cov must be a square matrix")
            dim = cov.shape[0]
        elif dim is None:
            raise ValueError("Must provide either cov or dim")

        self.dim = dim
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    @abstractmethod
    def get_covariance(self) -> Tensor:
        """Return Σ as a dense (dim, dim) matrix."""

    @abstractmethod
    def set_covariance(self, cov: Tensor) -> None:
        """Store Σ from a dense (dim, dim) matrix."""

    @abstractmethod
    def clone(self) -> "Covariance":
        """Return an independent copy."""

    @abstractmethod
    def log_det(self) -> Tensor:
        """Return log|Σ|."""

    @abstractmethod
    def inverse(self) -> Tensor:
        """Return Σ⁻¹."""

    @abstractmethod
    def solve(self, b: Tensor) -> Tensor:
        """Solve Σx = b."""

    @abstractmethod
    def matmul(self, x: Tensor) -> Tensor:
        """Compute Σx."""

    def trace(self) -> Tensor:
        """Return tr(Σ)."""
        return self.get_covariance().diagonal().sum()

    def num_parameters(self) -> int:
        """Number of free parameters in this representation."""
        return self.dim * (self.dim + 1) // 2

    def is_psd(self, tol: float = 1e-8) -> bool:
        """Check PSD."""
        eigvals = torch.linalg.eigvalsh(self.get_covariance())
        return bool((eigvals >= -tol).all())

    def validate(self, tol: float = 1e-8) -> None:
        """Raise if not PSD for debugging purposes."""
        if not self.is_psd(tol):
            eigvals = torch.linalg.eigvalsh(self.get_covariance())
            raise ValueError(f"Not PSD: min eigenvalue = {eigvals.min().item():.2e}")

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "Covariance":
        """Return a copy on the given device and dtype."""
        raise NotImplementedError


class DenseCovariance(Covariance):
    """Direct dense matrix storage."""

    def __init__(
        self,
        cov: Tensor = None,
        dim: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float64,
        init_scale: float = 1.0,
        loading: float = 0.0,
    ):
        """Store ``cov``, or ``init_scale`` times the identity."""
        super().__init__(cov=cov, dim=dim, device=device, dtype=dtype)
        self.loading = loading
        if cov is not None:
            cov = cov.to(device=self.device, dtype=self.dtype)
            self._cov = cov + loading * torch.eye(self.dim, device=self.device, dtype=self.dtype)
        else:
            self._cov = init_scale * torch.eye(self.dim, device=self.device, dtype=self.dtype)

    def get_covariance(self) -> Tensor:
        """Return the stored matrix."""
        return self._cov

    def set_covariance(self, cov: Tensor) -> None:
        """Replace the stored matrix."""
        self._cov = cov.to(device=self.device, dtype=self.dtype)

    def clone(self) -> "DenseCovariance":
        """Return an independent copy."""
        new = DenseCovariance(
            dim=self.dim,
            device=self.device,
            dtype=self.dtype,
            loading=self.loading,
        )
        new._cov = self._cov.clone()
        return new

    def log_det(self) -> Tensor:
        """Return log|Σ|."""
        return torch.linalg.slogdet(self._cov)[1]

    def inverse(self) -> Tensor:
        """Return Σ⁻¹."""
        return torch.linalg.inv(self._cov)

    def solve(self, b: Tensor) -> Tensor:
        """Solve Σx = b."""
        return torch.linalg.solve(self._cov, b)

    def matmul(self, x: Tensor) -> Tensor:
        """Compute Σx."""
        return self._cov @ x

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "DenseCovariance":
        """Return a copy on the given device and dtype."""
        new = DenseCovariance(
            dim=self.dim,
            device=device or self.device,
            dtype=dtype or self.dtype,
            loading=self.loading,
        )
        new._cov = self._cov.to(device=new.device, dtype=new.dtype)
        return new


class CholeskyCovariance(Covariance):
    """Cholesky-factor storage, Σ = LLᵀ: always PSD, stabler solves."""

    def __init__(
        self,
        cov: Tensor = None,
        dim: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float64,
        init_scale: float = 1.0,
        loading: float = 0.0,
    ):
        """Factor ``cov``, or ``init_scale`` times the identity."""
        super().__init__(cov=cov, dim=dim, device=device, dtype=dtype)
        self.loading = loading
        if cov is not None:
            cov = cov.to(device=self.device, dtype=self.dtype)
            # Loading applies to a supplied covariance, not to the default.
            cov = cov + loading * torch.eye(self.dim, device=self.device, dtype=self.dtype)
        else:
            cov = init_scale * torch.eye(self.dim, device=self.device, dtype=self.dtype)
        self.set_covariance(cov)

    def get_covariance(self) -> Tensor:
        """Return Σ = LLᵀ."""
        return self._L @ self._L.T

    def set_covariance(self, cov: Tensor) -> None:
        """Store Σ by factoring it."""
        cov = cov.to(device=self.device, dtype=self.dtype)
        self._L = torch.linalg.cholesky(cov)

    def set_cholesky(self, L: Tensor) -> None:
        """Directly set L when caller already has it."""
        self._L = L.to(device=self.device, dtype=self.dtype)

    def get_cholesky(self) -> Tensor:
        """Return the factor L."""
        return self._L

    def clone(self) -> "CholeskyCovariance":
        """Return an independent copy."""
        new = CholeskyCovariance(
            dim=self.dim,
            device=self.device,
            dtype=self.dtype,
            loading=self.loading,
        )
        new._L = self._L.clone()
        return new

    def log_det(self) -> Tensor:
        """Return log|Σ| = 2 Σ log diag(L)."""
        return 2.0 * self._L.diagonal().log().sum()

    def inverse(self) -> Tensor:
        """Return Σ⁻¹ by triangular solve."""
        I = torch.eye(self.dim, device=self.device, dtype=self.dtype)
        L_inv = torch.linalg.solve_triangular(self._L, I, upper=False)
        return L_inv.T @ L_inv

    def solve(self, b: Tensor) -> Tensor:
        """Solve Σx = b by forward and back substitution."""
        squeeze = b.ndim == 1
        if squeeze:
            b = b.unsqueeze(-1)
        y = torch.linalg.solve_triangular(self._L, b, upper=False)
        x = torch.linalg.solve_triangular(self._L.T, y, upper=True)
        return x.squeeze(-1) if squeeze else x

    def matmul(self, x: Tensor) -> Tensor:
        """Compute Σx."""
        return self._L @ (self._L.T @ x)

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "CholeskyCovariance":
        """Return a copy on the given device and dtype."""
        new = CholeskyCovariance(
            dim=self.dim,
            device=device or self.device,
            dtype=dtype or self.dtype,
            loading=self.loading,
        )
        new._L = self._L.to(device=new.device, dtype=new.dtype)
        return new


class DiagonalCovariance(Covariance):
    """Variances only, in O(d) storage.  Off-diagonal entries are discarded."""

    def __init__(
        self,
        cov: Tensor = None,
        dim: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float64,
        init_scale: float = 1.0,
        loading: float = 0.0,
    ):
        """Keep the diagonal of ``cov``, or ``init_scale`` for every variance."""
        super().__init__(cov=cov, dim=dim, device=device, dtype=dtype)
        self.loading = loading
        if cov is not None:
            cov = cov.to(device=self.device, dtype=self.dtype)
            self._var = cov.diagonal() + loading
        else:
            self._var = init_scale * torch.ones(self.dim, device=self.device, dtype=self.dtype)

    def get_covariance(self) -> Tensor:
        """Return full diagonal matrix for interface compatibility."""
        return torch.diag(self._var)

    def set_covariance(self, cov: Tensor) -> None:
        """Extract and store only diagonal elements."""
        cov = cov.to(device=self.device, dtype=self.dtype)
        self._var = cov.diagonal()

    def get_variance(self) -> Tensor:
        """Return the stored variance vector directly."""
        return self._var

    def set_variance(self, var: Tensor) -> None:
        """Directly set variance vector."""
        self._var = var.to(device=self.device, dtype=self.dtype)

    def clone(self) -> "DiagonalCovariance":
        """Return an independent copy."""
        new = DiagonalCovariance(
            dim=self.dim,
            device=self.device,
            dtype=self.dtype,
            loading=self.loading,
        )
        new._var = self._var.clone()
        return new

    def log_det(self) -> Tensor:
        """log|Σ| = sum of log of variances."""
        return self._var.log().sum()

    def inverse(self) -> Tensor:
        """Return diagonal matrix with 1/variances."""
        return torch.diag(1.0 / self._var)

    def solve(self, b: Tensor) -> Tensor:
        """Solve Σx = b via element-wise division."""
        if b.ndim == 1:
            return b / self._var
        else:
            # b is (..., dim) or (dim, k)
            return b / self._var.unsqueeze(-1) if b.shape[0] == self.dim else b / self._var

    def matmul(self, x: Tensor) -> Tensor:
        """Compute Σx via element-wise multiplication."""
        if x.ndim == 1:
            return self._var * x
        else:
            return self._var.unsqueeze(-1) * x if x.shape[0] == self.dim else self._var * x

    def trace(self) -> Tensor:
        """Sum of variances."""
        return self._var.sum()

    def num_parameters(self) -> int:
        """Diagonal has only d parameters, not d(d+1)/2."""
        return self.dim

    def is_psd(self, tol: float = 1e-8) -> bool:
        """PSD iff all variances are non-negative."""
        return bool((self._var >= -tol).all())

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "DiagonalCovariance":
        """Return a copy on the given device and dtype."""
        new = DiagonalCovariance(
            dim=self.dim,
            device=device or self.device,
            dtype=dtype or self.dtype,
            loading=self.loading,
        )
        new._var = self._var.to(device=new.device, dtype=new.dtype)
        return new
