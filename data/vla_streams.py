"""
VLA adaptation streams — online residual learning on a frozen pi0.5 policy.

Each of the 15 seeds gets one fixed stream of 300 (action, residual) pairs plus
256 separate calibration actions.  The actions were recorded once from the
nominal pi0.5 policy in LIBERO and are replayed here as plain arrays: nothing in
this module needs the policy, the simulator, or a GPU.

The learning problem is a *known* correction.  Every 60 pairs the control frame
yaws by a further 18 degrees (0, 18, 36, 54, 72), and the supervised target is

    y = R(theta) a - a

with ``R(theta)`` mixing translation x/y and orientation-vector x/y about z and
leaving z, orientation z and the gripper alone.  ``load`` recomputes the targets
from the actions and asserts they match what is stored, so a corrupted or
mislabelled stream fails loudly rather than training on quietly wrong labels.

At evaluation time the adapter's predicted residual is added to the policy's
action and the inverse rotation is applied before the environment sees it, so a
perfect correction recovers the nominal command.  The rotation is therefore an
exactly-known linear disturbance, not an unknown shift the learner must
discover.  Read that caveat before drawing adaptation conclusions from it.

Usage:
    stream = load(113001)
    X = stream.normalized_references()   # (300, 7) network inputs
    Y = stream.targets                   # (300, 7) physical-unit residuals
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "vla"
STREAM_DIR = DATA_DIR / "streams"
EPISODES = DATA_DIR / "episodes.csv.gz"

SEEDS = tuple(range(113001, 113016))
ACTION_DIM = 7
TRAIN_PAIRS = 300
CALIBRATION_ACTIONS = 256
REGIME_LENGTH = 60
FRAME_ANGLES_DEGREES = (0, 18, 36, 54, 72)

#: Dimensions the yaw acts on: translation x/y and orientation-vector x/y.
ROTATED_PAIRS = (0, 3)


def rotation(theta: float) -> np.ndarray:
    """The 7-D yaw used by the experiment, as a row-vector matrix.

    Applied as ``a @ rotation(theta).T``.  Only the two coordinate pairs in
    ``ROTATED_PAIRS`` move; dimension 2 (z), 5 (orientation z) and 6 (gripper)
    are left untouched.
    """
    matrix = np.eye(ACTION_DIM, dtype=np.float64)
    cosine, sine = np.cos(theta), np.sin(theta)
    for j in ROTATED_PAIRS:
        matrix[j, j] = cosine
        matrix[j, j + 1] = -sine
        matrix[j + 1, j] = sine
        matrix[j + 1, j + 1] = cosine
    return matrix


def theta_schedule() -> np.ndarray:
    """Per-sample yaw in radians: five regimes of ``REGIME_LENGTH`` pairs."""
    angles = np.deg2rad(np.asarray(FRAME_ANGLES_DEGREES, dtype=np.float64))
    return np.repeat(angles, REGIME_LENGTH)


@dataclass(frozen=True)
class VLAStream:
    """One seed's frozen adaptation stream."""

    seed: int
    references: np.ndarray               # (300, 7) nominal policy actions
    targets: np.ndarray                  # (300, 7) residuals, physical units
    theta: np.ndarray                    # (300,)   yaw in radians
    calibration_references: np.ndarray   # (256, 7) held out from training
    ref_mean: np.ndarray                 # (7,) calibration mean
    ref_std: np.ndarray                  # (7,) calibration population std
    source_files: tuple[str, ...]
    sample_indices: np.ndarray

    def normalized_references(self) -> np.ndarray:
        """Network inputs: calibration-standardised actions.

        Targets are deliberately *not* normalised — they stay in physical
        action units, so a model's meaning does not drift with a statistic.
        """
        return (self.references - self.ref_mean) / self.ref_std

    def regime_index(self) -> np.ndarray:
        """Which frame regime (0-4) each pair belongs to."""
        return np.arange(len(self.theta)) // REGIME_LENGTH


def load(seed: int, *, verify: bool = True) -> VLAStream:
    """Load one seed's stream.

    ``verify`` recomputes the targets and the calibration statistics from the
    stored actions and requires exact agreement.  Leave it on: it is the check
    that the labels are the documented closed-form ones.
    """
    if seed not in SEEDS:
        raise ValueError(f"Unknown VLA seed {seed}; expected one of {SEEDS[0]}..{SEEDS[-1]}")
    path = STREAM_DIR / f"{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing VLA stream {path}")

    with np.load(path, allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}

    shapes = {
        "references": (TRAIN_PAIRS, ACTION_DIM),
        "targets": (TRAIN_PAIRS, ACTION_DIM),
        "calibration_references": (CALIBRATION_ACTIONS, ACTION_DIM),
        "ref_mean": (ACTION_DIM,),
        "ref_std": (ACTION_DIM,),
        "theta": (TRAIN_PAIRS,),
    }
    for name, shape in shapes.items():
        value = arrays[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{seed}: {name} has shape {value.shape}, expected {shape}, "
                             f"or contains nonfinite values")

    if verify:
        _verify(seed, arrays)

    return VLAStream(
        seed=seed,
        references=arrays["references"].astype(np.float64),
        targets=arrays["targets"].astype(np.float64),
        theta=arrays["theta"].astype(np.float64),
        calibration_references=arrays["calibration_references"].astype(np.float64),
        ref_mean=arrays["ref_mean"].astype(np.float64),
        ref_std=arrays["ref_std"].astype(np.float64),
        source_files=tuple(arrays["source_files"].tolist()),
        sample_indices=arrays["sample_indices"],
    )


def load_all(*, verify: bool = True) -> dict[int, VLAStream]:
    """Every seed's stream, keyed by seed."""
    return {seed: load(seed, verify=verify) for seed in SEEDS}


def _verify(seed: int, arrays: dict[str, np.ndarray]) -> None:
    """Re-derive the schedule, the labels and the normalisation from scratch."""
    expected_theta = theta_schedule()
    if not np.array_equal(arrays["theta"], expected_theta):
        raise ValueError(f"{seed}: yaw schedule differs from {FRAME_ANGLES_DEGREES} degrees per "
                         f"{REGIME_LENGTH} pairs")

    expected_targets = np.stack([
        action @ rotation(theta).T - action
        for action, theta in zip(arrays["references"], arrays["theta"])
    ])
    if not np.array_equal(arrays["targets"], expected_targets):
        worst = float(np.abs(arrays["targets"] - expected_targets).max())
        raise ValueError(f"{seed}: stored targets are not R(theta)a - a "
                         f"(max |difference| {worst:g})")

    calibration = arrays["calibration_references"]
    if not np.array_equal(arrays["ref_mean"], calibration.mean(0)):
        raise ValueError(f"{seed}: ref_mean is not the calibration mean")
    std = calibration.std(0)
    if not np.array_equal(arrays["ref_std"], np.where(std > 1e-6, std, 1.0)):
        raise ValueError(f"{seed}: ref_std is not the calibration population std "
                         f"(constant dimensions set to 1)")


def load_episodes():
    """The archived rollout outcomes as a DataFrame, one row per episode.

    27,000 rows: 15 seeds x 4 methods x 30 evaluation points x 3 tasks x 5
    initial states.  These are measurements, not derived results — reproducing
    them needs the pi0.5 weights, LIBERO and a GPU — which is why they live
    under ``data/`` rather than ``out/``.
    """
    import pandas as pd

    if not EPISODES.exists():
        raise FileNotFoundError(f"Missing VLA episode table {EPISODES}")
    return pd.read_csv(EPISODES)
