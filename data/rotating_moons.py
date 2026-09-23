"""
Rotating moons — a non-stationary binary classification stream.

The classic two-moons dataset (sklearn) where the entire pattern rotates
slowly over time, producing smooth concept drift.  At each time step t in
[0, n_total) the moons are rotated by

    θ(t) = total_rotation * (t / n_total)

so the decision boundary sweeps continuously through the plane.

Abrupt drift — what the paper uses — replaces the linear sweep with
piecewise-constant segments: the angle is held fixed for ``segment_length`` samples, then
jumps to a new uniformly-random angle.  This produces sudden concept
drift at known change points, making it easy to see whether a learner
can recover after a distribution shift.

Implementation notes
--------------------
We generate one large moons batch up front, shuffle it, then rotate each
point by the angle corresponding to its position in the stream.  Because
every point in the shuffled batch is an iid draw from the base moons
distribution, rotating point i by θ(i) is equivalent to drawing one fresh
sample from the distribution rotated by θ(i).  This avoids the
per-sample ``make_moons`` call and guarantees balanced classes.

Standardisation is *isotropic*: we subtract the centroid and divide both
axes by the *same* scalar (the pooled standard deviation across both
features).  Anisotropic (per-axis) standardisation would assign different
scale factors to x and y, distorting the geometry when the pattern
rotates — circles become ellipses, moons get stretched.

Three public entry points:

    ``generate_stream``
        Returns the full (X, y) stream of training points with per-sample
        rotation applied and isotropic standardisation (smooth drift).

    ``generate_stream_abrupt``
        Same interface, but with piecewise-constant rotation angles that
        jump at regular intervals (abrupt drift).

    ``generate_test_set``
        Returns a fresh iid test set from the moons distribution at a
        given rotation angle, standardised with the same statistics used
        for the training stream.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons

# =============================================================================
# Low-level helpers
# =============================================================================

def _rotate(X: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate 2-D points around the origin by *angle_deg* degrees."""
    θ = np.radians(angle_deg)
    c, s = np.cos(θ), np.sin(θ)
    R = np.array([[c, -s], [s, c]])
    return X @ R.T


def _isotropic_standardise(
    X: np.ndarray,
    x_mean: np.ndarray,
    x_scale: float,
) -> np.ndarray:
    """Center and scale both axes by the same factor."""
    return (X - x_mean) / x_scale


def _generate_base_batch(n_total: int, noise: float,
                         rng: np.random.Generator):
    """Generate and shuffle a base moons batch (unrotated)."""
    X_base, y_base = make_moons(
        n_samples=n_total, noise=noise,
        random_state=rng.integers(2**31),
    )
    X_base = X_base.astype(np.float64)
    y_base = y_base.astype(np.float64)

    perm = rng.permutation(n_total)
    return X_base[perm], y_base[perm]


def _apply_rotation_and_standardise(
    X_base: np.ndarray,
    angles: np.ndarray,
    standardise_window: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate each sample by its angle, then isotropically standardise.

    Returns (X_standardised, x_mean, x_scale).
    """
    θ = np.radians(angles)
    cos_θ, sin_θ = np.cos(θ), np.sin(θ)
    X_rot = np.empty_like(X_base)
    X_rot[:, 0] = cos_θ * X_base[:, 0] - sin_θ * X_base[:, 1]
    X_rot[:, 1] = sin_θ * X_base[:, 0] + cos_θ * X_base[:, 1]

    window = X_rot[:standardise_window]
    x_mean = window.mean(axis=0)
    x_scale = float(window.std() + 1e-8)
    X_rot = _isotropic_standardise(X_rot, x_mean, x_scale)

    return X_rot, x_mean, x_scale


# =============================================================================
# Public API
# =============================================================================

def generate_stream(
    n_total: int = 2000,
    noise: float = 0.15,
    total_rotation: float = 180.0,
    seed: int = 7302519,
    standardise_window: int | None = None,
) -> dict:
    """Generate the full rotating-moons training stream (smooth drift).

    Each sample is drawn independently from a moons distribution whose
    rotation angle increases linearly from 0° to ``total_rotation``.

    Parameters
    ----------
    n_total : int
        Total number of stream samples.
    noise : float
        Moons crescent noise (passed to ``make_moons``).
    total_rotation : float
        Cumulative rotation (degrees) over the full stream.
    seed : int
        Random seed for reproducibility.
    standardise_window : int or None
        Number of initial samples used to compute standardisation
        statistics.  Defaults to ``n_total // 5``.

    Returns
    -------
    dict with keys:
        X        (n_total, 2)  float64 — standardised features
        y        (n_total,)    float64 — labels {0, 1}
        angles   (n_total,)    float64 — rotation angle per sample (degrees)
        x_mean   (2,)          float64 — centroid used for standardisation
        x_scale  float                 — isotropic scale factor
    """
    rng = np.random.default_rng(seed)

    if standardise_window is None:
        standardise_window = max(50, n_total // 5)

    X_base, y_base = _generate_base_batch(n_total, noise, rng)
    angles = np.linspace(0.0, total_rotation, n_total)
    X_rot, x_mean, x_scale = _apply_rotation_and_standardise(
        X_base, angles, standardise_window,
    )

    return {
        "X": X_rot,
        "y": y_base,
        "angles": angles,
        "x_mean": x_mean,
        "x_scale": x_scale,
    }


def generate_stream_abrupt(
    n_total: int = 2000,
    noise: float = 0.15,
    segment_length: int = 500,
    angle_range: float = 360.0,
    seed: int = 7302519,
    standardise_window: int | None = None,
) -> dict:
    """Generate a rotating-moons stream with abrupt concept drift.

    The rotation angle is piecewise constant: it stays fixed for
    ``segment_length`` samples, then jumps to a new angle drawn
    uniformly from [0, ``angle_range``).  The first segment always
    starts at 0°.

    Parameters
    ----------
    n_total : int
        Total number of stream samples.
    noise : float
        Moons crescent noise (passed to ``make_moons``).
    segment_length : int
        Number of samples between consecutive angle jumps.
    angle_range : float
        Angles are drawn uniformly from [0, angle_range).
    seed : int
        Random seed for reproducibility.
    standardise_window : int or None
        Number of initial samples used to compute standardisation
        statistics.  Defaults to ``n_total // 5``.

    Returns
    -------
    dict with keys:
        X               (n_total, 2)  float64 — standardised features
        y               (n_total,)    float64 — labels {0, 1}
        angles          (n_total,)    float64 — rotation angle per sample
        x_mean          (2,)          float64 — centroid for standardisation
        x_scale         float                 — isotropic scale factor
        change_points   list[int]             — stream indices where angle
                                                jumps (excluding t=0)
    """
    rng = np.random.default_rng(seed)

    if standardise_window is None:
        standardise_window = max(50, n_total // 5)

    X_base, y_base = _generate_base_batch(n_total, noise, rng)

    # ── Build piecewise-constant angle schedule ──────────────────────
    n_segments = max(1, (n_total + segment_length - 1) // segment_length)
    # First segment at 0°; remaining segments at random angles.
    segment_angles = np.empty(n_segments)
    segment_angles[0] = 0.0
    segment_angles[1:] = rng.uniform(0.0, angle_range, size=n_segments - 1)

    angles = np.empty(n_total, dtype=np.float64)
    change_points = []
    for seg in range(n_segments):
        start = seg * segment_length
        end = min(start + segment_length, n_total)
        angles[start:end] = segment_angles[seg]
        if seg > 0 and start < n_total:
            change_points.append(start)

    X_rot, x_mean, x_scale = _apply_rotation_and_standardise(
        X_base, angles, standardise_window,
    )

    return {
        "X": X_rot,
        "y": y_base,
        "angles": angles,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "change_points": change_points,
    }


def generate_test_set(
    angle_deg: float,
    n_test: int = 300,
    noise: float = 0.15,
    seed: int = 7302519,
    x_mean: np.ndarray | None = None,
    x_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a fresh iid test set at a specific rotation angle.

    If ``x_mean`` and ``x_scale`` are supplied, features are standardised
    with those statistics (should match the training stream).

    Returns (X_test, y_test) as float64 arrays.
    """
    rng = np.random.default_rng(seed)
    X, y = make_moons(n_samples=n_test, noise=noise,
                      random_state=rng.integers(2**31))
    X = _rotate(X.astype(np.float64), angle_deg)
    y = y.astype(np.float64)

    if x_mean is not None and x_scale is not None:
        X = _isotropic_standardise(X, x_mean, x_scale)

    return X, y
