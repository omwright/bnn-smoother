"""
Trajectory collection for online dynamics learning experiments.

Provides a uniform interface for collecting transition data from simulated
environments.  Each collector returns a dict with numpy arrays:

    {
        "states":  (T, d_state),      # s_t
        "actions": (T, d_action),      # a_t
        "deltas":  (T, d_state),       # Δs_t = s_{t+1} − s_t
    }

Learning the residual Δs (rather than the next state directly) is standard
practice — it makes the targets zero-mean and simplifies standardisation.

Supported environments
----------------------
cartpole
    Gymnasium CartPole-v1 (continuous force variant).
    State: (x, ẋ, θ, θ̇) — 4 dimensions.
    Action: continuous force ∈ [-10, 10] N.

industrial_benchmark
    Siemens Industrial Benchmark (IB).
    Observable: 6 variables (v, g, h, consumption, fatigue, cost).
    Input: sliding window of last L time steps → (6·L)-dim input.
    Action: (Δv, Δg, Δh) — 3 continuous steering deltas.
    Target: next-step observables (6-D or 3-D reward components).

Usage:
    from data.dynamics_envs import collect_cartpole, collect_industrial

    data = collect_cartpole(n_episodes=20, episode_length=200, seed=42)
    data = collect_industrial(n_steps=5000, state_window=5, seed=42)
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# CartPole (continuous force)
# =============================================================================

def _cartpole_step(state: np.ndarray, force: float, dt: float = 0.02):
    """One integration step for the CartPole dynamics.

    Uses the same physics as Gymnasium's CartPoleEnv but with continuous
    force input.  Euler integration with step size dt.

    State: (x, x_dot, theta, theta_dot).
    """
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5  # half-pole length
    polemass_length = masspole * length

    x, x_dot, theta, theta_dot = state

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    # Equations of motion, with continuous force.
    temp = (force + polemass_length * theta_dot**2 * sin_th) / total_mass
    theta_acc = (gravity * sin_th - cos_th * temp) / (
        length * (4.0 / 3.0 - masspole * cos_th**2 / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * cos_th / total_mass

    x_new = x + dt * x_dot
    x_dot_new = x_dot + dt * x_acc
    theta_new = theta + dt * theta_dot
    theta_dot_new = theta_dot + dt * theta_acc

    return np.array([x_new, x_dot_new, theta_new, theta_dot_new])


def collect_cartpole(
    n_episodes: int = 20,
    episode_length: int = 200,
    force_range: float = 10.0,
    seed: int = 42,
) -> dict:
    """Collect transition data from the continuous CartPole.

    Uses a random policy (uniform force) so the data explores the state
    space broadly.  The policy is fixed across methods — the comparison
    is purely on learning speed from the same stream.

    Returns
    -------
    dict with keys:
        states  : (T, 4) — s_t
        actions : (T, 1) — a_t (scalar force, stored as 2-D for consistency)
        deltas  : (T, 4) — s_{t+1} − s_t
        episode_starts : list[int] — index of first transition per episode
    """
    rng = np.random.default_rng(seed)

    all_states = []
    all_actions = []
    all_deltas = []
    episode_starts = []

    for ep in range(n_episodes):
        episode_starts.append(len(all_states))

        # Random initial state near the upright equilibrium.
        state = rng.uniform(-0.05, 0.05, size=4)

        for t in range(episode_length):
            force = rng.uniform(-force_range, force_range)
            next_state = _cartpole_step(state, force)

            all_states.append(state.copy())
            all_actions.append([force])
            all_deltas.append(next_state - state)

            state = next_state

    return {
        "states": np.array(all_states, dtype=np.float64),
        "actions": np.array(all_actions, dtype=np.float64),
        "deltas": np.array(all_deltas, dtype=np.float64),
        "episode_starts": episode_starts,
        "state_dim": 4,
        "action_dim": 1,
        "state_labels": ["x", "x_dot", "theta", "theta_dot"],
    }


# =============================================================================
# Industrial Benchmark
# =============================================================================

def _patch_and_import_ids():
    """Import the IB core simulator, patching numpy 2.0 incompatibilities.

    The ``industrialbenchmark-python`` package uses deprecated numpy
    aliases (``np.float``, ``np.int``, ``np.bool``) that were removed in
    NumPy 2.0.  We restore them as builtins before importing, then use
    the ``IDS`` simulator directly — bypassing ``IBGym`` which also
    depends on the deprecated ``gym`` package.
    """
    import numpy as _np

    for alias, builtin in [("float", float), ("int", int), ("bool", bool)]:
        if not hasattr(_np, alias):
            setattr(_np, alias, builtin)

    try:
        from industrial_benchmark_python.IDS import IDS
        return IDS
    except ImportError:
        raise ImportError(
            "Industrial Benchmark not installed.  Run:\n"
            "  uv pip install git+https://github.com/siemens/industrialbenchmark"
        )


def collect_industrial(
    n_steps: int = 5000,
    state_window: int = 5,
    seed: int = 42,
    target_mode: str = "full",
) -> dict:
    """Collect transition data from the Siemens Industrial Benchmark.

    The IB has 6 observable variables per time step:
        (velocity, gain, shift, consumption, fatigue, cost)

    The input at each step is a sliding window of the last `state_window`
    observations → (6 × state_window)-dimensional vector.

    The action is (Δvelocity, Δgain, Δshift) — 3 continuous deltas.

    Parameters
    ----------
    n_steps : int
        Number of transitions to collect.
    state_window : int
        Number of lagged observations to concatenate as input (L in the
        outline).  Default 5 → 30-D input.
    seed : int
        Random seed for the environment and policy.
    target_mode : str
        "full"   → predict all 6 next-step observables (default).
        "reward" → predict only (consumption, fatigue, cost) — the 3
                   stochastic reward components.

    Returns
    -------
    dict with keys:
        states  : (T', d_input)  — windowed observation vectors
        actions : (T', 3)        — action deltas
        deltas  : (T', d_target) — next-step target (residual)
    """
    IDS = _patch_and_import_ids()

    rng = np.random.default_rng(seed)

    # The core simulator, without the gym wrapper.
    ib = IDS(50, inital_seed=seed)

    # Of the IDS state dict, the 6 visible observables, as in Depeweg et al.
    # (2017): v (velocity), g (gain), h (shift), c (consumption), f (fatigue)
    # and cost.
    _OBS_KEYS = ["v", "g", "h", "c", "f", "cost"]

    def _read_obs() -> np.ndarray:
        """Read the current 6-D observable vector from the IDS state dict."""
        return np.array([ib.state[k] for k in _OBS_KEYS], dtype=np.float64)

    observations = []
    actions = []

    observations.append(_read_obs())

    total_needed = n_steps + state_window  # extra steps for windowing

    for _ in range(total_needed):
        # Random steering deltas in [-1, 1].
        action = rng.uniform(-1, 1, size=3)
        ib.step(action)

        observations.append(_read_obs())
        actions.append(action.copy())

    observations = np.array(observations)  # (total_needed + 1, 6)
    actions = np.array(actions)            # (total_needed, 3)

    windowed_states = []
    windowed_actions = []
    windowed_deltas = []

    for t in range(state_window, len(observations) - 1):
        # Input: observations from t-L+1 to t, concatenated.
        window = observations[t - state_window + 1: t + 1].flatten()
        windowed_states.append(window)
        windowed_actions.append(actions[t - 1])  # action taken at step t-1

        if target_mode == "reward":
            # Only the 3 stochastic components.
            delta = observations[t + 1, 3:6] - observations[t, 3:6]
        else:
            delta = observations[t + 1] - observations[t]
        windowed_deltas.append(delta)

    windowed_states = np.array(windowed_states[:n_steps], dtype=np.float64)
    windowed_actions = np.array(windowed_actions[:n_steps], dtype=np.float64)
    windowed_deltas = np.array(windowed_deltas[:n_steps], dtype=np.float64)

    d_target = windowed_deltas.shape[1]
    d_input = windowed_states.shape[1]

    target_labels = ["v", "g", "h", "consumption", "fatigue", "cost"]
    if target_mode == "reward":
        target_labels = ["consumption", "fatigue", "cost"]

    return {
        "states": windowed_states,
        "actions": windowed_actions,
        "deltas": windowed_deltas,
        "state_dim": d_input,
        "action_dim": 3,
        "target_dim": d_target,
        "state_labels": target_labels,
        "episode_starts": [0],  # single continuous run
        # Un-windowed, for the rollouts: observations[i] is the last time
        # step of states[i].
        "raw_observations": observations[state_window: state_window + n_steps].copy(),
        "state_window": state_window,
        "target_mode": target_mode,
    }


# =============================================================================
# Unified interface
# =============================================================================

ENVIRONMENTS = {
    "cartpole": collect_cartpole,
    "industrial_benchmark": collect_industrial,
}


def collect_transitions(environment: str, seed: int = 42, **kwargs) -> dict:
    """Collect transition data from a named environment.

    Parameters
    ----------
    environment : str
        One of: "cartpole", "industrial_benchmark".
    seed : int
        Random seed for reproducibility.
    **kwargs
        Environment-specific arguments (n_episodes, episode_length, etc.).

    Returns
    -------
    dict with at least: states, actions, deltas.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment {environment!r}. "
            f"Choose from: {sorted(ENVIRONMENTS)}"
        )
    return ENVIRONMENTS[environment](seed=seed, **kwargs)
