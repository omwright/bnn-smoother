import math
from typing import Tuple

import torch

EPS_ZERO = 1e-8  # Divide by zero
pi = math.pi
sqrt2 = math.sqrt(2.0)
sqrt2pi = math.sqrt(2.0 * pi)
one_ovr_sqrt2 = 1.0 / sqrt2
one_ovr_sqrt2pi = 1.0 / sqrt2pi
log2pi = math.log(2 * pi)

# Probit approximation constant: λ² = π/8
_LAMBDA_SQ = pi / 8.0


def standard_normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF, φ(x)."""
    return (1 / math.sqrt(2 * math.pi)) * torch.exp(-(x**2) / 2)


def standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, Φ(x) = ½(1 + erf(x/√2))."""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


# =============================================================================
# ReLU Moments
# =============================================================================


def relu_moment(m: torch.Tensor, s: torch.Tensor, min=EPS_ZERO) -> torch.Tensor:
    """Mean of ReLU(z), z ~ N(m, s²).  Shapes (*, n) in and out."""
    s_safe = torch.clamp(s, min=min)  # Avoid division by zero
    z = m / s_safe
    return m * standard_normal_cdf(z) + s_safe * standard_normal_pdf(z)


def relu_variance(m: torch.Tensor, s: torch.Tensor, min=EPS_ZERO) -> torch.Tensor:
    """Variance of ReLU(z), z ~ N(m, s²).  Shapes (*, n) in and out."""
    s_safe = torch.clamp(s, min=min)
    z = m / s_safe

    cdf = standard_normal_cdf(z)
    pdf = standard_normal_pdf(z)

    mean = m * cdf + s_safe * pdf
    second_moment = (m**2 + s_safe**2) * cdf + m * s_safe * pdf
    variance = second_moment - mean**2

    return torch.clamp(variance, min=0.0)


def relu_covariance(mu: torch.Tensor, cov: torch.Tensor, order: int = 4) -> torch.Tensor:
    """Full output covariance of ReLU(z), z ~ N(mu, Σ).

    Truncated Hadamard series of Wright et al. (2024).  With z = μ/σ, the
    derivatives of the marginal moment function are f⁽¹⁾ = Φ(z),
    f⁽²⁾ = φ(z)/σ, f⁽³⁾ = −z φ(z)/σ², f⁽⁴⁾ = (z²−1) φ(z)/σ³.

    mu is (batch, dim), cov is (batch, dim, dim), order is 1-4; returns
    (batch, dim, dim).
    """
    sigma = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))  # (batch, dim)
    z = mu / sigma

    p = standard_normal_pdf(z)
    P = standard_normal_cdf(z)

    f1 = P
    f2 = p / sigma
    f3 = -z * p / sigma**2
    f4 = (z**2 - 1) * p / sigma**3

    def batch_outer(a, b):
        """(batch, dim) x (batch, dim) -> (batch, dim, dim)."""
        return a.unsqueeze(-1) * b.unsqueeze(-2)

    C2 = cov * cov  # Hadamard square

    cov_out = batch_outer(f1, f1) * cov
    if order >= 2:
        cov_out = cov_out + batch_outer(f2, f2) * C2 / 2
    if order >= 3:
        cov_out = cov_out + batch_outer(f3, f3) * (C2 * cov) / 6
    if order >= 4:
        cov_out = cov_out + batch_outer(f4, f4) * (C2 * C2) / 24

    # Overwrite the diagonal with the exact closed-form marginal variance.
    z_mean = mu * P + sigma * p
    z_var = (mu**2 + sigma**2) * P + mu * sigma * p - z_mean**2
    torch.diagonal(cov_out, dim1=-2, dim2=-1).copy_(z_var)

    return cov_out


def relu_moments(
    m: torch.Tensor, s: torch.Tensor, min=EPS_ZERO
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of ReLU(z), z ~ N(m, s²)."""
    s_safe = torch.clamp(s, min=min)
    z = m / s_safe

    cdf = standard_normal_cdf(z)
    pdf = standard_normal_pdf(z)

    mean = m * cdf + s_safe * pdf
    second_moment = (m**2 + s_safe**2) * cdf + m * s_safe * pdf
    variance = torch.clamp(second_moment - mean**2, min=0.0)

    return mean, variance


# =============================================================================
# Approximated Sigmoid Moments
# =============================================================================
#
# The probit approximation (MacKay 1992, Barber & Bishop 1998), as in
# Appendix A of Wagner et al. (2023, arXiv:2110.00944).  The full output
# covariance uses the Hadamard series of Wright et al. (2024, arXiv:2403.16163),
# Appendix B.


def sigmoid_moment(m: torch.Tensor, s: torch.Tensor, min=EPS_ZERO) -> torch.Tensor:
    """Mean of σ(z), z ~ N(m, s²): E[σ(z)] ≈ σ(m/t), t = √(1 + (π/8)s²)."""
    s_safe = torch.clamp(s, min=min)
    t = torch.sqrt(1.0 + _LAMBDA_SQ * s_safe**2)
    return torch.sigmoid(m / t)


def sigmoid_variance(m: torch.Tensor, s: torch.Tensor, min=EPS_ZERO) -> torch.Tensor:
    """Variance of σ(z), z ~ N(m, s²): ≈ σ(m/t)(1 − σ(m/t))(1 − 1/t)."""
    s_safe = torch.clamp(s, min=min)
    t = torch.sqrt(1.0 + _LAMBDA_SQ * s_safe**2)
    sig = torch.sigmoid(m / t)
    variance = sig * (1.0 - sig) * (1.0 - 1.0 / t)
    return torch.clamp(variance, min=0.0)


def sigmoid_moments(
    m: torch.Tensor, s: torch.Tensor, min=EPS_ZERO
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and variance of σ(z), z ~ N(m, s²)."""
    s_safe = torch.clamp(s, min=min)
    t = torch.sqrt(1.0 + _LAMBDA_SQ * s_safe**2)
    sig = torch.sigmoid(m / t)
    variance = torch.clamp(sig * (1.0 - sig) * (1.0 - 1.0 / t), min=0.0)
    return sig, variance


def sigmoid_covariance(
    mu: torch.Tensor, cov: torch.Tensor, order: int = 4
) -> torch.Tensor:
    """Full output covariance of σ(z), z ~ N(mu, Σ).

    Truncated Hadamard series applied to the probit-approximated sigmoid,
    f(μ) = σ(μ/t) with t = √(1 + (π/8)σ²).  With h = σ(μ/t):

        f⁽¹⁾ = h(1−h)/t                      f⁽²⁾ = h(1−h)(1−2h)/t²
        f⁽³⁾ = h(1−h)(1 − 6h(1−h))/t³        f⁽⁴⁾ = h(1−h)(1−2h)(1 − 12h(1−h))/t⁴

    mu is (batch, dim), cov is (batch, dim, dim), order is 1-4; returns
    (batch, dim, dim).
    """
    sigma_sq = torch.diagonal(cov, dim1=-2, dim2=-1)  # (batch, dim)
    t = torch.sqrt(1.0 + _LAMBDA_SQ * sigma_sq)       # (batch, dim)
    h = torch.sigmoid(mu / t)                          # (batch, dim)

    h1h = h * (1.0 - h)  # h(1−h), reused across all derivatives

    f1 = h1h / t
    f2 = h1h * (1.0 - 2.0 * h) / (t**2)
    f3 = h1h * (1.0 - 6.0 * h1h) / (t**3)
    f4 = h1h * (1.0 - 2.0 * h) * (1.0 - 12.0 * h1h) / (t**4)

    def batch_outer(a, b):
        """(batch, dim) x (batch, dim) -> (batch, dim, dim)."""
        return a.unsqueeze(-1) * b.unsqueeze(-2)

    C2 = cov * cov  # Hadamard square

    cov_out = batch_outer(f1, f1) * cov
    if order >= 2:
        cov_out = cov_out + batch_outer(f2, f2) * C2 / 2
    if order >= 3:
        cov_out = cov_out + batch_outer(f3, f3) * (C2 * cov) / 6
    if order >= 4:
        cov_out = cov_out + batch_outer(f4, f4) * (C2 * C2) / 24

    # Exact diagonal: overwrite with closed-form marginal variance
    diag_var = h1h * (1.0 - 1.0 / t)
    torch.diagonal(cov_out, dim1=-2, dim2=-1).copy_(
        torch.clamp(diag_var, min=0.0)
    )

    return cov_out
