"""
Online dynamics learning experiment.

Evaluates how well BNN methods learn a simulator's transition function
f(s, a) → Δs in an online (single-pass, no-replay) setting.  This is the
dynamics-learning component of model-based RL, isolated from policy
optimisation so that model quality is measured on its own.

Supported environments:
    cartpole             — 4-D state, 1-D action, 4-D target
    industrial_benchmark — 30-D windowed input, 3-D action, 6-D target

Per trial: collect trajectories under a random policy, split them temporally,
standardize on the training statistics, then stream the transitions one at a
time, evaluating on the held-out set at log-spaced checkpoints and rolling out
multi-step trajectories at the end.

Metrics: one-step RMSE and NLL (diagonal, plus full for methods with
``forward``), rollout RMSE at each horizon, and ms per update.

Usage:
    uv run python run_dynamics_experiment.py --config configs/fig2_cartpole.yaml
    uv run python run_dynamics_experiment.py --config configs/fig3_industrial.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import yaml

from bnn import resolve_device
from data.dynamics_envs import collect_transitions
from methods import build
from utils.streaming import (
    extract_weights,
    inject_weights,
    is_kalman,
    update_single,
)

# =============================================================================
# Data preparation
# =============================================================================

def prepare_dynamics_data(raw: dict, test_fraction: float = 0.2, device: torch.device = None):
    """Build standardised train/test tensors from raw transitions.

    The split is temporal -- the last ``test_fraction`` -- because later
    transitions may come from different parts of the state space.  The input
    is concat(state, action) and the target is the next-state delta.
    """
    states = raw["states"]    # (T, d_state)
    actions = raw["actions"]  # (T, d_action)
    deltas = raw["deltas"]    # (T, d_target)

    T = states.shape[0]
    n_test = max(1, int(T * test_fraction))
    n_train = T - n_test

    X = np.concatenate([states, actions], axis=1)  # (T, d_input)
    Y = deltas                                      # (T, d_target)

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    # Standardise on training statistics only.
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    y_mean = Y_train.mean(axis=0)
    y_std = Y_train.std(axis=0) + 1e-8

    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    Y_train = (Y_train - y_mean) / y_std
    Y_test = (Y_test - y_mean) / y_std

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float64, device=device),
        "Y_train": torch.tensor(Y_train, dtype=torch.float64, device=device),
        "X_test": torch.tensor(X_test, dtype=torch.float64, device=device),
        "Y_test": torch.tensor(Y_test, dtype=torch.float64, device=device),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,       # (d_target,) ndarray — needed for de-standardisation
        "n_train": n_train,
        "n_test": n_test,
        # Raw, un-standardised, in temporal order, for the rollouts.
        "raw_states": raw["states"],
        "raw_actions": raw["actions"],
        "raw_deltas": raw["deltas"],
        "episode_starts": raw.get("episode_starts", [0]),
        # Industrial Benchmark: un-windowed observations for the rollouts.
        "raw_observations": raw.get("raw_observations", None),
        "state_window": raw.get("state_window", None),
    }


# =============================================================================
# Evaluation schedule
# =============================================================================

def make_eval_schedule(n_train: int, n_points: int) -> list[int]:
    """Log-spaced checkpoint indices in [1, n_train], including both endpoints."""
    if n_train <= 1:
        return [1]
    pts = np.logspace(0, np.log10(n_train), n_points)
    pts = np.unique(np.round(pts).astype(int))
    pts = pts[(pts >= 1) & (pts <= n_train)]
    return sorted(set(pts.tolist()) | {n_train})


# =============================================================================
# One-step evaluation
# =============================================================================

_LN2PI = 1.8378770664093453


def evaluate_one_step(method, X_test, Y_test, y_std, method_spec=None):
    """One-step test-set RMSE and NLL, in the original scale."""
    Y_pred, Y_var = method.predict(X_test)
    d_target = Y_test.shape[1]
    y_std_t = torch.tensor(y_std, dtype=Y_pred.dtype, device=Y_pred.device)

    diff = Y_test.to(Y_pred.dtype) - Y_pred
    rmse_per_dim = (diff.pow(2).mean(dim=0).sqrt() * y_std_t).tolist()
    rmse = float(np.mean(rmse_per_dim))

    var = torch.clamp(Y_var, min=1e-8)
    nll_per_sample = 0.5 * (
        torch.log(2 * torch.pi * var) + diff.pow(2) / var
    ).sum(dim=-1)
    log_jacobian = float(np.log(y_std).sum())
    nll_diag = float(nll_per_sample.mean().item()) + log_jacobian

    # Methods without a full predictive covariance have no forward().
    nll_full = float("nan")
    if hasattr(method, "forward"):
        try:
            nlls = []
            for i in range(X_test.shape[0]):
                mean, cov = method.forward(X_test[i])
                reg = cov + 1e-8 * torch.eye(d_target, dtype=cov.dtype, device=cov.device)
                d = Y_test[i].to(mean.dtype) - mean
                _, logdet = torch.linalg.slogdet(reg)
                quad = d @ torch.linalg.solve(reg, d)
                nlls.append(0.5 * (logdet + quad + d_target * _LN2PI))
            nll_full = float(torch.stack(nlls).mean().item()) + log_jacobian
        except Exception:
            pass

    return {
        "rmse": rmse,
        "rmse_per_dim": rmse_per_dim,
        "nll_diag": nll_diag,
        "nll_full": nll_full,
    }


# =============================================================================
# Multi-step rollout evaluation
# =============================================================================

def evaluate_rollouts(
    method,
    split: dict,
    horizons: list[int],
    n_trajectories: int = 50,
    seed: int = 42,
) -> dict:
    """Multi-step deterministic rollout error, per horizon.

    Flat-state (CartPole): s_{h+1} = s_h + predict(concat(s_h, a_h)).
    Windowed (Industrial Benchmark): keep the last L raw observations, rebuild
    the window each step, predict the 6-D delta, update the buffer.
    """
    raw_states = split["raw_states"]
    raw_actions = split["raw_actions"]
    x_mean = split["x_mean"]
    x_std = split["x_std"]
    y_mean = split["y_mean"]
    y_std = split["y_std"]
    n_train = split["n_train"]
    ref = split["X_test"]   # rollout inputs must match the training dtype

    raw_obs = split.get("raw_observations", None)
    state_window = split.get("state_window", None)
    windowed = raw_obs is not None and state_window is not None

    max_H = max(horizons)

    rng = np.random.default_rng(seed)
    valid_starts = list(range(n_train, raw_states.shape[0] - max_H))
    if not valid_starts:
        return {h: {"rmse": float("nan"), "rmse_per_dim": []} for h in horizons}

    n_traj = min(n_trajectories, len(valid_starts))
    start_indices = rng.choice(valid_starts, size=n_traj, replace=False)

    results = {}
    for H in horizons:
        errors = []

        for s0_idx in start_indices:
            if s0_idx + H >= raw_states.shape[0]:
                continue

            if windowed:
                # IB mode: maintain a buffer of raw observations.
                obs_dim = raw_obs.shape[1]  # 6
                # Build initial window buffer from the raw observations.
                buf = []
                for k in range(state_window):
                    # raw_obs[i] is raw_states[i]'s last time step.
                    obs_idx = s0_idx - state_window + 1 + k
                    if obs_idx < 0:
                        buf.append(np.zeros(obs_dim))
                    else:
                        buf.append(raw_obs[obs_idx].copy())

                for h in range(H):
                    a = raw_actions[s0_idx + h]
                    window = np.concatenate(buf)
                    inp = np.concatenate([window, a])
                    inp_std = (inp - x_mean) / x_std
                    inp_t = torch.tensor(inp_std, dtype=ref.dtype, device=ref.device).unsqueeze(0)
                    delta_pred, _ = method.predict(inp_t)
                    delta_pred = delta_pred.squeeze(0).detach().cpu().numpy()
                    delta_orig = delta_pred * y_std + y_mean

                    # New observation = last observation + delta.
                    new_obs = buf[-1] + delta_orig[:obs_dim]
                    buf.pop(0)
                    buf.append(new_obs)

                true_obs = raw_obs[s0_idx + H] if s0_idx + H < len(raw_obs) else None
                if true_obs is not None:
                    errors.append((buf[-1] - true_obs) ** 2)
            else:
                # Flat-state mode (CartPole).
                s = raw_states[s0_idx].copy()
                d_state = s.shape[0]

                for h in range(H):
                    a = raw_actions[s0_idx + h]
                    inp = np.concatenate([s, a])
                    inp_std = (inp - x_mean) / x_std
                    inp_t = torch.tensor(inp_std, dtype=ref.dtype, device=ref.device).unsqueeze(0)
                    delta_pred, _ = method.predict(inp_t)
                    delta_pred = delta_pred.squeeze(0).detach().cpu().numpy()
                    delta_orig = delta_pred * y_std + y_mean
                    s = s + delta_orig[:d_state]

                true_state = raw_states[s0_idx + H]
                errors.append((s - true_state) ** 2)

        if errors:
            errors = np.array(errors)
            rmse_per_dim = np.sqrt(errors.mean(axis=0)).tolist()
            rmse = float(np.mean(rmse_per_dim))
        else:
            rmse_per_dim = []
            rmse = float("nan")

        results[H] = {"rmse": rmse, "rmse_per_dim": rmse_per_dim}

    return results


# =============================================================================
# Config helpers
# =============================================================================

def parse_method_specs(cfg: dict) -> list[dict]:
    """Extract method specifications from config."""
    specs = []
    for entry in cfg["methods"]:
        specs.append({
            "name": entry["name"],
            "model": entry.get("model", {}),
            "label": entry.get("label", entry["name"]),
        })
    return specs


def display_name(spec: dict) -> str:
    """Legend label for a method spec."""
    return spec.get("label", spec["name"])


# =============================================================================
# Core streaming loop (one trial)
# =============================================================================

def _sync(device: torch.device) -> None:
    """Timing helper for CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_one_trial(
    method_specs: list[dict],
    split: dict,
    eval_schedule: list[int],
    seed: int,
    share_init: bool = True,
    device: torch.device = None,
) -> list[dict]:
    """Run all methods on one data stream.  Returns one result dict per method."""
    device = torch.device("cpu") if device is None else device
    X_train = split["X_train"]
    Y_train = split["Y_train"]
    X_test = split["X_test"]
    Y_test = split["Y_test"]
    y_std = split["y_std"]
    n_train = X_train.shape[0]
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    eval_set = set(eval_schedule)

    methods = []
    for spec in method_specs:
        torch.manual_seed(seed)
        m = build(
            spec["name"],
            input_dim=input_dim,
            output_dim=output_dim,
            config=dict(spec["model"]),
            device=device,
        )
        methods.append(m)

    if share_init:
        kalman_idx = [i for i, s in enumerate(method_specs) if is_kalman(s)]
        if len(kalman_idx) >= 2:
            canonical = extract_weights(methods[kalman_idx[0]])
            for idx in kalman_idx[1:]:
                inject_weights(methods[idx], canonical)

    results = []
    for spec in method_specs:
        results.append({
            "method": display_name(spec),
            "trace": [],
            "update_times": [],
        })

    # t = 0: the prior's predictions.
    for i, method in enumerate(methods):
        metrics = evaluate_one_step(method, X_test, Y_test, y_std)
        metrics["t"] = 0
        results[i]["trace"].append(metrics)

    for t in range(1, n_train + 1):
        x = X_train[t - 1]
        y = Y_train[t - 1]

        for i, method in enumerate(methods):
            _sync(device)
            t0 = time.perf_counter()
            update_single(method, x, y)
            _sync(device)
            results[i]["update_times"].append(time.perf_counter() - t0)

        if t in eval_set:
            for i, method in enumerate(methods):
                metrics = evaluate_one_step(method, X_test, Y_test, y_std)
                metrics["t"] = t
                results[i]["trace"].append(metrics)

    return results


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_trials(
    all_trial_results: list[list[dict]],
    method_specs: list[dict],
) -> dict:
    """Aggregate per-trial traces into mean ± SE curves."""
    n_methods = len(method_specs)
    n_trials = len(all_trial_results)
    metrics = ["rmse", "nll_diag", "nll_full"]

    agg = {}
    for mi in range(n_methods):
        label = display_name(method_specs[mi])
        traces = [all_trial_results[tr][mi]["trace"] for tr in range(n_trials)]
        t_values = [pt["t"] for pt in traces[0]]
        n_pts = len(t_values)

        arrays = {}
        for m in metrics:
            arr = np.full((n_trials, n_pts), np.nan)
            for tr in range(n_trials):
                for p, pt in enumerate(traces[tr]):
                    arr[tr, p] = pt.get(m, float("nan"))
            arrays[m] = arr

        se_denom = np.sqrt(n_trials)
        entry = {"t": t_values}
        for m in metrics:
            with np.errstate(all="ignore"):
                entry[f"{m}_mean"] = np.nanmean(arrays[m], axis=0).tolist()
                entry[f"{m}_se"] = (np.nanstd(arrays[m], axis=0) / se_denom).tolist()

        all_times = []
        for tr in range(n_trials):
            all_times.extend(all_trial_results[tr][mi]["update_times"])
        entry["ms_per_update_mean"] = float(np.mean(all_times) * 1e3)
        entry["ms_per_update_std"] = float(np.std(all_times) * 1e3)

        agg[label] = entry

    return agg


def aggregate_rollouts(
    all_rollouts: list[dict[str, dict]],
    method_specs: list[dict],
) -> dict:
    """Aggregate rollouts into {label: {horizon: {rmse_mean, rmse_se}}}."""
    n_trials = len(all_rollouts)
    se_denom = np.sqrt(n_trials)
    agg = {}

    for mi, spec in enumerate(method_specs):
        label = display_name(spec)
        horizons = sorted(all_rollouts[0][label].keys())
        entry = {}
        for H in horizons:
            vals = [all_rollouts[tr][label][H]["rmse"] for tr in range(n_trials)]
            entry[H] = {
                "rmse_mean": float(np.nanmean(vals)),
                "rmse_se": float(np.nanstd(vals) / se_denom),
            }
        agg[label] = entry

    return agg


# =============================================================================
# Plotting
# =============================================================================

def plot_learning_curves(agg: dict, env_name: str, metric: str, out_path: str):
    """Plot one metric vs. samples seen."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    label_map = {
        "rmse": "One-step RMSE",
        "nll_diag": "One-step NLL (diagonal)",
        "nll_full": "One-step NLL (full cov)",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Colour by position in the config's method list.
    for index, (method_name, data) in enumerate(agg.items()):
        t = np.array(data["t"])
        mean = np.array(data[f"{metric}_mean"])
        se = np.array(data[f"{metric}_se"])

        if np.all(np.isnan(mean)):
            continue

        colour = f"C{index % 10}"
        ax.plot(t, mean, label=method_name, linewidth=1.5, color=colour)
        ax.fill_between(t, mean - se, mean + se, alpha=0.2, color=colour)

    ax.set_xlabel("Training transitions seen")
    ax.set_ylabel(label_map.get(metric, metric))
    ax.set_title(f"Online Dynamics — {env_name} — {label_map.get(metric, metric)}")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nll_combined(agg: dict, env_name: str, out_path: str):
    """Plot NLL learning curves with all methods on one axes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_plotted = False

    for index, (method_name, data) in enumerate(agg.items()):
        t = np.array(data["t"])

        # Prefer full NLL; fall back to diagonal.
        full_mean = np.array(data["nll_full_mean"])
        diag_mean = np.array(data["nll_diag_mean"])

        if not np.all(np.isnan(full_mean)):
            mean = full_mean
            se = np.array(data["nll_full_se"])
            suffix = " (full)"
        elif not np.all(np.isnan(diag_mean)):
            mean = diag_mean
            se = np.array(data["nll_diag_se"])
            suffix = " (diag)"
        else:
            continue  # no NLL for this method (e.g. SGD)

        colour = f"C{index % 10}"   # by config position; see plot_learning_curves
        ax.plot(t, mean, label=method_name + suffix, linewidth=1.5, color=colour)
        ax.fill_between(t, mean - se, mean + se, alpha=0.2, color=colour)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Training transitions seen")
    ax.set_ylabel("One-step Test NLL")
    ax.set_title(f"Online Dynamics — {env_name} — Test NLL")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_rmse(rollout_agg: dict, env_name: str, out_path: str):
    """Plot rollout RMSE vs. horizon for each method."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for index, (method_name, data) in enumerate(rollout_agg.items()):
        horizons = sorted(data.keys())
        means = [data[h]["rmse_mean"] for h in horizons]
        ses = [data[h]["rmse_se"] for h in horizons]
        means = np.array(means)
        ses = np.array(ses)

        ax.errorbar(horizons, means, yerr=ses, label=method_name,
                     marker="o", capsize=3, linewidth=1.5,
                     color=f"C{index % 10}")   # see plot_learning_curves

    ax.set_xlabel("Rollout horizon H")
    ax.set_ylabel("Trajectory RMSE")
    ax.set_title(f"Multi-step Rollout — {env_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Display
# =============================================================================

def _fmt(val, se):
    """Format a metric as 'mean ± se', or 'N/A' if NaN."""
    if np.isnan(val):
        return f"{'N/A':>16}"
    return f"{val:>7.3f} ± {se:<6.3f}"


def print_summary(agg: dict, rollout_agg: dict, method_specs: list[dict]):
    """Print a final summary table with mean ± SE."""
    labels = [display_name(s) for s in method_specs]
    w = max(len(l) for l in labels) + 2

    print("\n" + "=" * 100)
    print("FINAL ONE-STEP METRICS (after seeing all training data)")
    print("=" * 100)
    print(f"  {'Method':<{w}} {'RMSE':>16}  {'NLL (diag)':>16}  "
          f"{'NLL (full)':>16}  {'ms/update':>10}")
    print(f"  {'-' * (w + 66)}")

    for label in labels:
        d = agg[label]
        rmse_str = _fmt(d["rmse_mean"][-1], d["rmse_se"][-1])
        nll_d_str = _fmt(d["nll_diag_mean"][-1], d["nll_diag_se"][-1])
        nll_f_str = _fmt(d["nll_full_mean"][-1], d["nll_full_se"][-1])
        ms = d["ms_per_update_mean"]
        print(f"  {label:<{w}} {rmse_str}  {nll_d_str}  {nll_f_str}  {ms:>8.3f}")

    if rollout_agg:
        horizons = sorted(next(iter(rollout_agg.values())).keys())
        print("\n" + "=" * 100)
        print("MULTI-STEP ROLLOUT RMSE")
        print("=" * 100)
        h_hdr = "  ".join(f"{'H=' + str(h):>16}" for h in horizons)
        print(f"  {'Method':<{w}}   {h_hdr}")
        print(f"  {'-' * (w + 2 + len(horizons) * 18)}")

        for label in labels:
            parts = []
            for h in horizons:
                r = rollout_agg[label][h]
                parts.append(_fmt(r["rmse_mean"], r["rmse_se"]))
            print(f"  {label:<{w}}   {'  '.join(parts)}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the configured trials, then aggregate, report and save."""
    parser = argparse.ArgumentParser(
        description="Online dynamics learning experiment")
    parser.add_argument("--config", type=str,
                        default="configs/fig2_cartpole.yaml")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-rollout", action="store_true",
                        help="skip multi-step rollout evaluation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    environment = cfg["environment"]
    seed = cfg.get("seed", 42)
    n_trials = cfg.get("n_trials", 5)
    n_eval_points = cfg.get("n_eval_points", 50)
    test_fraction = cfg.get("test_fraction", 0.2)
    share_init = cfg.get("share_init", True)
    device = resolve_device(cfg.get("device"))
    rollout_horizons = cfg.get("rollout_horizons", [1, 5, 10, 20])
    n_rollout_traj = cfg.get("n_rollout_trajectories", 50)

    method_specs = parse_method_specs(cfg)
    method_labels = [display_name(s) for s in method_specs]

    env_kwargs = {}
    for key in ["n_episodes", "episode_length", "force_range",
                 "n_steps", "state_window", "target_mode"]:
        if key in cfg:
            env_kwargs[key] = cfg[key]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"out/dynamics_{environment}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"Online Dynamics Learning — {environment}")
    print("=" * 70)
    print(f"  Methods:    {', '.join(method_labels)}")
    print(f"  Trials:     {n_trials} | seed: {seed}")
    print(f"  Test frac:  {test_fraction}")
    print(f"  Eval pts:   {n_eval_points} (log-spaced)")
    print(f"  Rollout H:  {rollout_horizons}")
    print(f"  Share init: {share_init}")
    print(f"  Device:     {device}")
    print(f"  Output:     {out_dir}/")
    print()

    all_trial_results = []
    all_rollout_results = []
    failures = 0

    for trial in range(n_trials):
        trial_seed = seed + trial * 1000
        print(f"  Trial {trial + 1}/{n_trials} (seed={trial_seed})...",
              end=" ", flush=True)

        raw = collect_transitions(environment, seed=trial_seed, **env_kwargs)
        split = prepare_dynamics_data(raw, test_fraction=test_fraction, device=device)

        n_train = split["n_train"]
        input_dim = split["X_train"].shape[1]
        output_dim = split["Y_train"].shape[1]
        if trial == 0:
            print(f"(T={raw['states'].shape[0]}, "
                  f"input_dim={input_dim}, output_dim={output_dim}, "
                  f"train={n_train})")

        eval_schedule = make_eval_schedule(n_train, n_eval_points)

        try:
            t0 = time.perf_counter()
            trial_results = run_one_trial(
                method_specs=method_specs,
                split=split,
                eval_schedule=eval_schedule,
                seed=trial_seed,
                share_init=share_init,
                device=device,
            )
            elapsed = time.perf_counter() - t0
            all_trial_results.append(trial_results)

            parts = []
            for r in trial_results:
                final_rmse = r["trace"][-1]["rmse"]
                parts.append(f"{r['method']}={final_rmse:.4f}")
            print(f"  done ({elapsed:.1f}s)  final RMSE: {', '.join(parts)}")

            if not args.no_rollout:
                rollout_for_trial = _run_rollout_trial(
                    method_specs, split, trial_seed, share_init,
                    rollout_horizons, n_rollout_traj, device=device,
                )
                all_rollout_results.append(rollout_for_trial)

        except Exception as e:
            failures += 1
            print(f"  FAILED ({e.__class__.__name__}: {e})")

    if failures:
        print(f"\n  {failures}/{n_trials} trial(s) failed due to numerical issues.")

    if not all_trial_results:
        print("\n  All trials failed — no results to aggregate.")
        return

    agg = aggregate_trials(all_trial_results, method_specs)

    rollout_agg = {}
    if all_rollout_results:
        rollout_agg = aggregate_rollouts(all_rollout_results, method_specs)

    print_summary(agg, rollout_agg, method_specs)

    if not args.no_plot:
        plot_dir = os.path.join(out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        for metric in ["rmse"]:
            plot_learning_curves(
                agg, environment, metric,
                os.path.join(plot_dir, f"{metric}.png"),
            )

        plot_nll_combined(
            agg, environment,
            os.path.join(plot_dir, "nll.png"),
        )

        if rollout_agg:
            plot_rollout_rmse(
                rollout_agg, environment,
                os.path.join(plot_dir, "rollout_rmse.png"),
            )

        print(f"\n  Plots saved to {plot_dir}/")

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        # Record the resolved device: `auto` differs per machine.
        yaml.dump({**cfg, "device": str(device)}, f)

    with open(os.path.join(out_dir, "aggregated.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # Per-trial traces, without the update times.  Every method sees the same
    # stream, so these support paired comparisons; the aggregate mean ± SE is
    # unpaired and hides the trial-to-trial variance common to all methods.
    traces = []
    for trial_results in all_trial_results:
        traces.append([
            {"method": r["method"], "trace": r["trace"]}
            for r in trial_results
        ])
    with open(os.path.join(out_dir, "traces.json"), "w") as f:
        json.dump(traces, f, indent=2)

    if rollout_agg:
        # JSON keys must be strings.
        rollout_json = {}
        for label, data in rollout_agg.items():
            rollout_json[label] = {str(h): v for h, v in data.items()}
        with open(os.path.join(out_dir, "rollout.json"), "w") as f:
            json.dump(rollout_json, f, indent=2)

        # Per-trial rollouts, paired for the same reason as traces.json: every
        # method rolls out from the identical start indices within a trial.
        rollout_traces = [
            {
                label: {str(h): v for h, v in data.items()}
                for label, data in trial.items()
            }
            for trial in all_rollout_results
        ]
        with open(os.path.join(out_dir, "rollout_traces.json"), "w") as f:
            json.dump(rollout_traces, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


# =============================================================================
# Rollout helper
# =============================================================================

@torch.no_grad()
def _run_rollout_trial(
    method_specs: list[dict],
    split: dict,
    seed: int,
    share_init: bool,
    horizons: list[int],
    n_traj: int,
    device: torch.device = None,
) -> dict:
    """Train every method, then roll out: {label: {horizon: {"rmse": ...}}}.

    Kept separate from ``run_one_trial`` so the trained methods are not held in
    memory alongside it.
    """
    X_train = split["X_train"]
    Y_train = split["Y_train"]
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    methods = []
    for spec in method_specs:
        torch.manual_seed(seed)
        m = build(spec["name"], input_dim=input_dim, output_dim=output_dim,
                  config=dict(spec["model"]), device=device)
        methods.append(m)

    if share_init:
        kalman_idx = [i for i, s in enumerate(method_specs) if is_kalman(s)]
        if len(kalman_idx) >= 2:
            canonical = extract_weights(methods[kalman_idx[0]])
            for idx in kalman_idx[1:]:
                inject_weights(methods[idx], canonical)

    for t in range(X_train.shape[0]):
        x, y = X_train[t], Y_train[t]
        for method in methods:
            update_single(method, x, y)

    results = {}
    for i, spec in enumerate(method_specs):
        label = display_name(spec)
        results[label] = evaluate_rollouts(
            methods[i], split, horizons, n_trajectories=n_traj, seed=seed,
        )

    return results


if __name__ == "__main__":
    main()
