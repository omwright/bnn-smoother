"""
Non-stationary adaptation: Online Rotating Moons.

The moons pattern jumps to a new random angle every ``segment_length``
samples, giving sudden concept drift at known change points.  Setting
``drift_type: smooth`` instead rotates the pattern linearly over the stream.

Produces:
    - Accuracy and BCE learning curves (mean ± SE across trials)
    - A snapshot plot showing decision boundaries at key rotation angles
    - JSON results for downstream analysis

Usage:
    uv run python run_online_moons_experiment.py --config configs/fig1_moons.yaml
    uv run python run_online_moons_experiment.py --config configs/fig1_moons.yaml --no-plot
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import yaml

from data.rotating_moons import (
    generate_stream,
    generate_stream_abrupt,
    generate_test_set,
)
from methods import build
from utils.streaming import (
    extract_weights,
    inject_weights,
    is_kalman,
    update_single,
)

# =============================================================================
# Evaluation
# =============================================================================

def evaluate_on_current_test(
    method,
    angle_deg: float,
    n_test: int,
    noise: float,
    x_mean: np.ndarray,
    x_scale: float,
    seed: int,
    method_spec: dict,
) -> dict:
    """Accuracy and BCE on a fresh test set at the current rotation angle."""
    X_test, y_test = generate_test_set(
        angle_deg=angle_deg,
        n_test=n_test,
        noise=noise,
        seed=seed,
        x_mean=x_mean,
        x_scale=x_scale,
    )
    X_t = torch.tensor(X_test, dtype=torch.float64)
    Y_t = torch.tensor(y_test, dtype=torch.float64).reshape(-1, 1)

    Y_pred, Y_var = method.predict(X_t)

    sigmoid_out = method_spec.get("model", {}).get("output_activation") == "sigmoid"
    if sigmoid_out:
        Y_prob = torch.clamp(Y_pred, min=1e-7, max=1.0 - 1e-7)
    else:
        Y_prob = torch.sigmoid(Y_pred)

    preds = (Y_prob >= 0.5).float()
    accuracy = float((preds == Y_t).float().mean().item())

    p = torch.clamp(Y_prob, min=1e-7, max=1.0 - 1e-7)
    bce = -(Y_t * torch.log(p) + (1.0 - Y_t) * torch.log(1.0 - p))
    bce = float(bce.mean().item())

    return {"accuracy": accuracy, "bce": bce}


# =============================================================================
# Evaluation schedule
# =============================================================================

def make_eval_schedule(n_total: int, n_points: int) -> list[int]:
    """Evenly-spaced checkpoints in [0, n_total], inclusive.

    Linear rather than log spacing: the drift is uniform in time, so the
    resolution should be too.
    """
    pts = np.linspace(0, n_total, n_points + 1, dtype=int)
    return sorted(set(pts.tolist()))


# =============================================================================
# Core loop for one trial
# =============================================================================

@torch.no_grad()
def run_one_trial(
    method_specs: list[dict],
    stream: dict,
    eval_schedule: list[int],
    n_test: int,
    noise: float,
    seed: int,
    share_init: bool,
) -> list[dict]:
    """Run all methods on one stream realisation.

    Returns one result dict per method, each containing the full metric trace.
    """
    X_train = torch.tensor(stream["X"], dtype=torch.float64)
    y_train = torch.tensor(stream["y"], dtype=torch.float64).reshape(-1, 1)
    angles = stream["angles"]
    x_mean = stream["x_mean"]
    x_scale = stream["x_scale"]
    n_total = X_train.shape[0]

    eval_set = set(eval_schedule)

    methods = []
    for spec in method_specs:
        torch.manual_seed(seed)
        m = build(
            spec["name"],
            input_dim=2,
            output_dim=1,
            config=dict(spec["model"]),
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
        label = spec.get("label", spec["name"])
        results.append({
            "method": label,
            "trace": [],
            "update_times": [],
        })

    # t = 0: the prior's predictions.
    if 0 in eval_set:
        for i, method in enumerate(methods):
            metrics = evaluate_on_current_test(
                method, angle_deg=angles[0], n_test=n_test, noise=noise,
                x_mean=x_mean, x_scale=x_scale, seed=seed + 100_000,
                method_spec=method_specs[i],
            )
            metrics["t"] = 0
            metrics["angle"] = float(angles[0])
            results[i]["trace"].append(metrics)

    for t in range(n_total):
        x = X_train[t]
        y = y_train[t]

        for i, method in enumerate(methods):
            t0 = time.perf_counter()
            update_single(method, x, y)
            results[i]["update_times"].append(time.perf_counter() - t0)

        samples_seen = t + 1
        if samples_seen in eval_set:
            current_angle = float(angles[t])
            for i, method in enumerate(methods):
                metrics = evaluate_on_current_test(
                    method, angle_deg=current_angle, n_test=n_test,
                    noise=noise, x_mean=x_mean, x_scale=x_scale,
                    seed=seed + 100_000 + samples_seen,
                    method_spec=method_specs[i],
                )
                metrics["t"] = samples_seen
                metrics["angle"] = current_angle
                results[i]["trace"].append(metrics)

    return results


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_trials(
    all_trial_results: list[list[dict]],
    method_specs: list[dict],
) -> dict:
    """Aggregate per-trial traces into mean ± SE curves.

    Returns dict mapping method label → {t, angle, accuracy_mean, accuracy_se,
    bce_mean, bce_se, ms_per_update_mean}.
    """
    n_methods = len(method_specs)
    n_trials = len(all_trial_results)
    metrics = ["accuracy", "bce"]

    agg = {}
    for mi in range(n_methods):
        label = method_specs[mi].get("label", method_specs[mi]["name"])

        traces = [all_trial_results[tr][mi]["trace"] for tr in range(n_trials)]
        t_values = [pt["t"] for pt in traces[0]]
        angle_values = [pt["angle"] for pt in traces[0]]
        n_pts = len(t_values)

        arrays = {}
        for m in metrics:
            arr = np.full((n_trials, n_pts), np.nan)
            for tr in range(n_trials):
                for p, pt in enumerate(traces[tr]):
                    arr[tr, p] = pt.get(m, float("nan"))
            arrays[m] = arr

        se_denom = np.sqrt(n_trials)
        entry = {"t": t_values, "angle": angle_values}
        for m in metrics:
            entry[f"{m}_mean"] = np.nanmean(arrays[m], axis=0).tolist()
            entry[f"{m}_se"] = (np.nanstd(arrays[m], axis=0) / se_denom).tolist()

        all_times = []
        for tr in range(n_trials):
            all_times.extend(all_trial_results[tr][mi]["update_times"])
        entry["ms_per_update_mean"] = float(np.mean(all_times) * 1e3)

        agg[label] = entry

    return agg


# =============================================================================
# Plotting
# =============================================================================

def plot_learning_curves(agg: dict, metric: str, out_path: str,
                         total_rotation: float | None = None,
                         change_points: list[int] | None = None):
    """Plot one metric over time with optional change-point markers.

    For smooth drift, a secondary x-axis shows the rotation angle.
    For abrupt drift, vertical dashed lines mark the change points.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

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

    ax.set_xlabel("Training samples seen")
    label_map = {"accuracy": "Accuracy (on current test set)",
                 "bce": "BCE (on current test set)"}
    ax.set_ylabel(label_map.get(metric, metric))

    drift_tag = "Abrupt" if change_points else "Smooth"
    ax.set_title(f"Online Rotating Moons ({drift_tag}) — "
                 f"{label_map.get(metric, metric)}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if change_points:
        for cp in change_points:
            ax.axvline(cp, color="grey", linestyle="--", linewidth=0.8,
                       alpha=0.6)
        # One legend entry for the group.
        ax.axvline(cp, color="grey", linestyle="--", linewidth=0.8,
                   alpha=0.6, label="Change point")
        ax.legend()

    elif total_rotation is not None:
        # Smooth drift: a secondary axis showing the rotation angle.
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        n_ticks = 5
        t_max = ax.get_xlim()[1]
        tick_positions = np.linspace(0, t_max, n_ticks + 1)
        tick_labels = [f"{total_rotation * tp / t_max:.0f}°"
                       for tp in tick_positions]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels)
        ax2.set_xlabel("Rotation angle")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decision_boundaries(
    method_specs: list[dict],
    methods_at_checkpoints: dict,
    stream: dict,
    checkpoint_angles: list[float],
    out_path: str,
    grid_res: int = 80,
):
    """Plot decision boundaries at a few key rotation angles.

    ``methods_at_checkpoints`` maps angle → list of (method, spec) pairs.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_angles = len(checkpoint_angles)
    n_methods = len(method_specs)
    fig, axes = plt.subplots(
        n_methods, n_angles,
        figsize=(4.5 * n_angles, 4 * n_methods),
        squeeze=False,
    )

    x_mean, x_scale = stream["x_mean"], stream["x_scale"]

    for col, angle in enumerate(checkpoint_angles):
        # Grid in standardised space, padded because rotation shifts points.
        pad = 4.0
        xx, yy = np.meshgrid(
            np.linspace(-pad, pad, grid_res),
            np.linspace(-pad, pad, grid_res),
        )
        grid = torch.tensor(
            np.c_[xx.ravel(), yy.ravel()], dtype=torch.float64
        )

        # Reference points to scatter over the boundary.
        X_ref, y_ref = generate_test_set(
            angle, n_test=200, noise=stream.get("noise", 0.15),
            seed=12345, x_mean=x_mean, x_scale=x_scale,
        )

        method_list = methods_at_checkpoints[angle]
        for row, (method, spec) in enumerate(method_list):
            ax = axes[row, col]
            label = spec.get("label", spec["name"])

            g_mean, g_var = method.predict(grid)
            sigmoid_out = spec.get("model", {}).get(
                "output_activation") == "sigmoid"
            if sigmoid_out:
                g_prob = torch.clamp(g_mean, 0.0, 1.0).numpy().reshape(xx.shape)
            else:
                g_prob = torch.sigmoid(g_mean).numpy().reshape(xx.shape)

            ax.contourf(
                xx, yy, g_prob,
                levels=np.linspace(0, 1, 21),
                cmap="RdYlBu_r", alpha=0.85,
            )
            ax.contour(xx, yy, g_prob, levels=[0.5],
                       colors="k", linewidths=1.5)

            for cls, marker, color in [(0, "o", "royalblue"),
                                       (1, "s", "firebrick")]:
                mask = y_ref == cls
                ax.scatter(
                    X_ref[mask, 0], X_ref[mask, 1],
                    c=color, marker=marker, s=10, alpha=0.5,
                    edgecolors="none",
                )

            ax.set_xlim(-pad, pad)
            ax.set_ylim(-pad, pad)
            ax.set_aspect("equal")
            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            if row == 0:
                ax.set_title(f"{angle:.0f}°", fontsize=11)

    fig.suptitle("Decision boundaries at key rotation angles", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Stream generation dispatch
# =============================================================================

def build_stream(stream_cfg: dict, seed: int) -> dict:
    """Build a training stream from the config, dispatching on drift_type."""
    drift_type = stream_cfg.get("drift_type", "smooth")

    if drift_type == "smooth":
        stream = generate_stream(
            n_total=stream_cfg["n_total"],
            noise=stream_cfg["noise"],
            total_rotation=stream_cfg["total_rotation"],
            seed=seed,
        )
    elif drift_type == "abrupt":
        stream = generate_stream_abrupt(
            n_total=stream_cfg["n_total"],
            noise=stream_cfg["noise"],
            segment_length=stream_cfg["segment_length"],
            angle_range=stream_cfg.get("angle_range", 360.0),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown drift_type: {drift_type!r}")

    stream["noise"] = stream_cfg["noise"]
    return stream


# =============================================================================
# Snapshot angle selection
# =============================================================================

def _pick_snapshot_angles(stream_cfg: dict, stream: dict,
                          n_total: int) -> list[float]:
    """Choose rotation angles at which to snapshot decision boundaries.

    Smooth: evenly spaced fractions of total_rotation (original behaviour).
    Abrupt: midpoint of each segment (one snapshot per regime).
    """
    drift_type = stream_cfg.get("drift_type", "smooth")

    if drift_type == "abrupt":
        seg_len = stream_cfg["segment_length"]
        angles_arr = stream["angles"]
        cps = [0] + stream.get("change_points", [])
        snapshot_angles = []
        for cp in cps:
            mid = min(cp + seg_len // 2, n_total - 1)
            snapshot_angles.append(float(angles_arr[mid]))
        return snapshot_angles
    else:
        total_rotation = stream_cfg["total_rotation"]
        return [0.0, total_rotation / 3, 2 * total_rotation / 3,
                total_rotation]


def _snapshot_index_for_angle(target_angle: float, stream: dict,
                              n_total: int) -> int:
    """Find the stream index whose angle is closest to target_angle."""
    angles_arr = stream["angles"]
    idx = int(np.argmin(np.abs(angles_arr - target_angle)))
    return idx


# =============================================================================
# Display
# =============================================================================

def print_summary(agg: dict, method_specs: list[dict]):
    """Print a final summary table."""
    print("\n" + "=" * 80)
    print("FINAL METRICS (at end of stream, evaluated on current distribution)")
    print("=" * 80)

    labels = [s.get("label", s["name"]) for s in method_specs]
    w = max(len(l) for l in labels) + 2

    print(f"  {'Method':<{w}} {'Accuracy':>16}  {'BCE':>16}  {'ms/update':>10}")
    print(f"  {'-' * (w + 48)}")

    for label in labels:
        d = agg[label]
        acc_m = d["accuracy_mean"][-1]
        acc_se = d["accuracy_se"][-1]
        bce_m = d["bce_mean"][-1]
        bce_se = d["bce_se"][-1]
        ms = d["ms_per_update_mean"]
        print(f"  {label:<{w}} "
              f"{acc_m:>6.1%} ± {acc_se:<6.1%}  "
              f"{bce_m:>6.3f} ± {bce_se:<6.3f}  "
              f"{ms:>8.3f}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the configured trials, then aggregate, report and save."""
    parser = argparse.ArgumentParser(
        description="Non-stationary adaptation: Online Rotating Moons")
    parser.add_argument("--config", type=str,
                        default="configs/fig1_moons.yaml")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stream_cfg = cfg["stream"]
    n_total = stream_cfg["n_total"]
    noise = stream_cfg["noise"]
    n_test = stream_cfg["n_test"]
    drift_type = stream_cfg.get("drift_type", "smooth")

    total_rotation = stream_cfg.get("total_rotation", None)   # smooth drift
    segment_length = stream_cfg.get("segment_length", None)   # abrupt drift

    seed = cfg.get("seed", 42)
    n_trials = cfg.get("n_trials", 5)
    n_eval_points = cfg.get("n_eval_points", 60)
    share_init = cfg.get("share_init", True)

    method_specs = cfg["methods"]
    method_labels = [s.get("label", s["name"]) for s in method_specs]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"out/online_moons_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("Non-stationary Adaptation: Online Rotating Moons")
    print("=" * 70)
    if drift_type == "smooth":
        print(f"  Drift:      smooth ({total_rotation}° linear rotation)")
    else:
        angle_range = stream_cfg.get("angle_range", 360.0)
        print(f"  Drift:      abrupt (jump every {segment_length} samples, "
              f"U[0, {angle_range}°))")
    print(f"  Stream:     {n_total} samples, noise={noise}")
    print(f"  Test set:   {n_test} samples (regenerated per checkpoint)")
    print(f"  Methods:    {', '.join(method_labels)}")
    print(f"  Trials:     {n_trials} | seed: {seed}")
    print(f"  Eval pts:   {n_eval_points} (linearly spaced)")
    print(f"  Share init: {share_init}")
    print(f"  Output:     {out_dir}/")
    print()

    eval_schedule = make_eval_schedule(n_total, n_eval_points)

    all_trial_results = []
    # The change-point positions are the same in every trial, only the angles
    # differ, so the first trial's are what the plots need.
    first_change_points = None

    for trial in range(n_trials):
        trial_seed = seed + trial * 1000
        print(f"  Trial {trial + 1}/{n_trials} (seed={trial_seed})...",
              end=" ", flush=True)

        stream = build_stream(stream_cfg, seed=trial_seed)

        if first_change_points is None:
            first_change_points = stream.get("change_points", None)

        t0 = time.perf_counter()
        trial_results = run_one_trial(
            method_specs=method_specs,
            stream=stream,
            eval_schedule=eval_schedule,
            n_test=n_test,
            noise=noise,
            seed=trial_seed,
            share_init=share_init,
        )
        elapsed = time.perf_counter() - t0
        all_trial_results.append(trial_results)

        parts = []
        for r in trial_results:
            final_acc = r["trace"][-1]["accuracy"]
            parts.append(f"{r['method']}={final_acc:.1%}")
        print(f"done ({elapsed:.1f}s)  final acc: {', '.join(parts)}")

    agg = aggregate_trials(all_trial_results, method_specs)
    print_summary(agg, method_specs)

    if not args.no_plot:
        plot_dir = os.path.join(out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        for metric in ["accuracy", "bce"]:
            plot_learning_curves(
                agg, metric,
                os.path.join(plot_dir, f"{metric}.png"),
                total_rotation=total_rotation,
                change_points=first_change_points,
            )
        print(f"\n  Learning curve plots saved to {plot_dir}/")

        # Decision-boundary snapshots need one more run, at the base seed.
        stream = build_stream(stream_cfg, seed=seed)

        snapshot_angles = _pick_snapshot_angles(stream_cfg, stream, n_total)

        snapshot_indices = set()
        for angle in snapshot_angles:
            idx = _snapshot_index_for_angle(angle, stream, n_total)
            snapshot_indices.add(min(idx, n_total))

        # Build once and snapshot along the way.
        X_train = torch.tensor(stream["X"], dtype=torch.float64)
        y_train = torch.tensor(stream["y"], dtype=torch.float64).reshape(-1, 1)
        angles = stream["angles"]

        torch_methods = []
        for spec in method_specs:
            torch.manual_seed(seed)
            m = build(spec["name"], input_dim=2, output_dim=1,
                      config=dict(spec["model"]))
            torch_methods.append(m)

        if share_init:
            kalman_idx = [i for i, s in enumerate(method_specs)
                          if is_kalman(s)]
            if len(kalman_idx) >= 2:
                canonical = extract_weights(torch_methods[kalman_idx[0]])
                for idx in kalman_idx[1:]:
                    inject_weights(torch_methods[idx], canonical)

        methods_at_checkpoints = {}

        for t in range(n_total):
            x = X_train[t]
            y = y_train[t]
            for m in torch_methods:
                update_single(m, x, y)

            if (t + 1) in snapshot_indices:
                angle = float(angles[t])
                methods_at_checkpoints[angle] = [
                    (copy.deepcopy(m), spec)
                    for m, spec in zip(torch_methods, method_specs)
                ]

        if 0 in snapshot_indices and 0.0 not in methods_at_checkpoints:
            fresh = []
            for spec in method_specs:
                torch.manual_seed(seed)
                m = build(spec["name"], input_dim=2, output_dim=1,
                          config=dict(spec["model"]))
                fresh.append(m)
            methods_at_checkpoints[0.0] = [
                (m, spec) for m, spec in zip(fresh, method_specs)
            ]

        available_angles = sorted(methods_at_checkpoints.keys())
        if available_angles:
            plot_decision_boundaries(
                method_specs, methods_at_checkpoints, stream,
                available_angles,
                os.path.join(plot_dir, "decision_boundaries.png"),
            )
            print(f"  Decision boundary plot saved to {plot_dir}/")

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    with open(os.path.join(out_dir, "aggregated.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # Per-trial traces, without the update times: every method sees the same
    # stream, so these support paired comparisons that the aggregate cannot.
    traces = []
    for trial_results in all_trial_results:
        traces.append([
            {"method": r["method"], "trace": r["trace"]}
            for r in trial_results
        ])
    with open(os.path.join(out_dir, "traces.json"), "w") as f:
        json.dump(traces, f, indent=2)

    print(f"\n  Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
