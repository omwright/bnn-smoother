#!/usr/bin/env python3
"""
VLA online adaptation experiment — paper Figure 4.

Trains a residual adapter on top of a frozen pi0.5 LIBERO policy as the control
frame yaws, then reports success rate against archived rollout outcomes.

    train    replay each seed's 300-pair stream for every method
    score    join adapters to episode outcomes, compute curves and summaries
    plot     render the success-rate figure

Scoring reads the archived episodes under data/vla/, so it describes the frozen
models; fresh rollouts need the separate project under vla/rollout/.

    uv run python run_vla_experiment.py --config configs/fig4_vla.yaml
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
from data.vla_streams import load as load_stream
from data.vla_streams import load_episodes
from methods import build

# This repository's method names vs the frozen archive's.
ARCHIVE_NAMES = {
    "kbnn_onestep": "kbnn_single",
    "wagner_kbnn": "wagner_kbnn",
    "tagi": "tagi",
    "replay_nn": "nn_adamw_single",
}
#: Offset applied to a seed to get the replay buffer's sampling generator.
BATCH_SEED_OFFSET = 2_000_003


# =============================================================================
# Model construction
# =============================================================================

def parse_method_specs(cfg: dict) -> list[dict]:
    """Extract method specifications from config."""
    return [
        {
            "name": entry["name"],
            "model": entry.get("model", {}),
            "label": entry.get("label", entry["name"]),
            "init": entry.get("init", "own"),
        }
        for entry in cfg["methods"]
    ]


def display_name(spec: dict) -> str:
    """Legend label for a method spec."""
    return spec.get("label", spec["name"])


def _zero_output_layer(name: str, net) -> None:
    """Zero the output layer, so an untrained adapter emits no residual."""
    if name == "kbnn_onestep":
        net.layers[-1].weight_mean.zero_()
    elif name == "wagner_kbnn":
        net.mw[-1].zero_()
    elif name == "tagi":
        net.mu_w[-1].zero_()
        net.mu_b[-1].zero_()
    elif name == "replay_nn":
        pass  # zero_init_output handles it at construction
    else:
        raise ValueError(f"No zero-output rule for method {name!r}")


def _copy_proposed_means(name: str, net, proposed_net) -> None:
    """Start a baseline from the proposed method's weight means."""
    if name == "wagner_kbnn":
        for i, layer in enumerate(proposed_net.layers):
            net.mw[i] = layer.weight_mean.clone()
    elif name == "tagi":
        for i, layer in enumerate(proposed_net.layers):
            # TAGI stores biases separately; the others use a final weight row.
            net.mu_w[i] = layer.weight_mean[:-1].clone()
            net.mu_b[i] = layer.weight_mean[-1].clone()
    else:
        raise ValueError(f"Cannot copy proposed means into {name!r}")


def build_model(spec: dict, cfg: dict, seed: int, device: torch.device):
    """Construct one method at one seed.

    Seeding happens here because ``matched_proposed_means`` builds the proposed
    method first and then rewinds the RNG, so the baseline draws what it would
    have drawn alone.
    """
    name = spec["name"]
    model_cfg = dict(spec["model"])
    if name == "replay_nn":
        model_cfg.setdefault("batch_seed", seed + BATCH_SEED_OFFSET)

    proposed_net = None
    if spec["init"] == "matched_proposed_means":
        base = next(s for s in parse_method_specs(cfg) if s["name"] == "kbnn_onestep")
        torch.manual_seed(seed)
        proposed = build("kbnn_onestep", input_dim=cfg["input_dim"],
                         output_dim=cfg["output_dim"], config=dict(base["model"]),
                         device=device)
        _zero_output_layer("kbnn_onestep", proposed.net)
        proposed_net = proposed.net
    elif spec["init"] not in ("own", "native"):
        raise ValueError(f"Unknown init {spec['init']!r} for method {name!r}")

    torch.manual_seed(seed)
    method = build(name, input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
                   config=model_cfg, device=device)
    if proposed_net is not None:
        _copy_proposed_means(name, method.net, proposed_net)
    _zero_output_layer(name, method.net)
    return method


def training_state(name: str, net) -> dict:
    """The complete numerical state, in the archive's field layout."""
    if name == "kbnn_onestep":
        return {"weight_means": [layer.weight_mean for layer in net.layers],
                "weight_covs": [layer._cov.get_covariance() for layer in net.layers]}
    if name == "wagner_kbnn":
        return {"mw": net.mw, "Cw": net.Cw}
    if name == "tagi":
        return {"mu_w": net.mu_w, "var_w": net.var_w, "mu_b": net.mu_b, "var_b": net.var_b}
    if name == "replay_nn":
        return net.training_state()
    raise ValueError(f"No state layout for method {name!r}")


# =============================================================================
# Training
# =============================================================================

def _sync(device: torch.device) -> None:
    """Wait for queued CUDA work, so update timings measure real work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def train_seed(spec, cfg, seed, device, checkpoints, out_dir=None, probes=None,
               tolerance=1e-6):
    """Replay one seed's stream for one method, checkpointing as configured."""
    name = spec["name"]
    stream = load_stream(seed)
    X = torch.tensor(stream.normalized_references(), dtype=torch.float64, device=device)
    Y = torch.tensor(stream.targets, dtype=torch.float64, device=device)
    ref_mean = torch.tensor(stream.ref_mean, dtype=torch.float64)
    ref_std = torch.tensor(stream.ref_std, dtype=torch.float64)

    method = build_model(spec, cfg, seed, device)
    net = method.net

    probe_x = None
    if probes is not None:
        probe_x = torch.tensor(probes["inputs"][str(seed)], dtype=torch.float64)

    times, worst_probe = [], 0.0
    for step in range(1, cfg["train_pairs"] + 1):
        _sync(device)
        started = time.perf_counter()
        net.update(X[step - 1], Y[step - 1])
        _sync(device)
        times.append(time.perf_counter() - started)

        if step not in checkpoints:
            continue
        state = training_state(name, net)
        payload = {"method": ARCHIVE_NAMES[name], "seed": seed, "samples_seen": step,
                   "config": dict(spec["model"]), "ref_mean": ref_mean,
                   "ref_std": ref_std, **state}
        relative = f"{seed}/{ARCHIVE_NAMES[name]}/step_{step:03d}.pt"

        if probe_x is not None:
            got = torch.stack([method.predict(row.unsqueeze(0))[0].squeeze(0).double()
                               for row in probe_x])
            want = torch.tensor(probes["predictions"][relative], dtype=torch.float64)
            error = float((got.cpu() - want).abs().max())
            worst_probe = max(worst_probe, error)
            if error > tolerance:
                raise SystemExit(
                    f"{relative}: prediction differs from the archived probe by "
                    f"{error:.3g}, above the {tolerance:g} tolerance.  Expected if "
                    f"you changed the config; drop --verify-probes if so."
                )

        if out_dir is not None:
            path = os.path.join(out_dir, "adapters", str(seed), name)
            os.makedirs(path, exist_ok=True)
            torch.save(payload, os.path.join(path, f"step_{step:03d}.pt"))

    return {"seed": seed, "method": name, "updates": cfg["train_pairs"],
            "ms_per_update_mean": float(np.mean(times) * 1000),
            "ms_per_update_median": float(np.median(times) * 1000),
            "max_probe_error": worst_probe if probes is not None else None}


# =============================================================================
# Scoring
# =============================================================================

def score(cfg: dict) -> dict:
    """Curves and summary statistics from the archived episode outcomes."""
    frame = load_episodes()
    specs = parse_method_specs(cfg)
    seeds, checkpoints = cfg["seeds"], cfg["checkpoints"]
    per_episode = cfg["episodes_per_task"] * len(cfg["tasks"])

    # rate[method]: (n_seeds, n_checkpoints) success rates
    rate = {}
    for spec in specs:
        archive = ARCHIVE_NAMES[spec["name"]]
        subset = frame[frame.method == archive]
        grid = np.full((len(seeds), len(checkpoints)), np.nan)
        grouped = subset.groupby(["seed", "samples_seen"])["success"].agg(["sum", "count"])
        for (seed, step), row in grouped.iterrows():
            if seed in seeds and step in checkpoints:
                if row["count"] != per_episode:
                    raise ValueError(
                        f"{archive} seed {seed} step {step}: {row['count']} episodes, "
                        f"expected {per_episode}"
                    )
                grid[seeds.index(seed), checkpoints.index(step)] = row["sum"] / row["count"]
        if not np.isfinite(grid).all():
            missing = int((~np.isfinite(grid)).sum())
            raise ValueError(f"{archive}: {missing} missing (seed, checkpoint) cells")
        rate[spec["name"]] = grid

    boot = cfg["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    draws = rng.integers(0, len(seeds), size=(boot["draws"], len(seeds)))
    low_q, high_q = [100 * q for q in boot["interval"]]

    curves = []
    for spec in specs:
        grid = rate[spec["name"]]
        for j, step in enumerate(checkpoints):
            column = grid[:, j]
            resampled = column[draws].mean(axis=1)
            curves.append({
                "method": spec["name"], "label": spec["label"], "samples_seen": step,
                "mean": float(column.mean()),
                "ci_low": float(np.percentile(resampled, low_q)),
                "ci_high": float(np.percentile(resampled, high_q)),
            })

    summary = {"seeds": seeds, "episodes": int(len(frame)),
               "checkpoints": checkpoints, "methods": {}, "contrasts": {}}
    per_seed = {}
    for spec in specs:
        values = rate[spec["name"]].mean(axis=1)
        per_seed[spec["name"]] = values
        resampled = values[draws].mean(axis=1)
        summary["methods"][spec["name"]] = {
            "label": spec["label"], "mean": float(values.mean()),
            "per_seed": values.tolist(),
            "ci95": [float(np.percentile(resampled, low_q)),
                     float(np.percentile(resampled, high_q))],
        }
    # Paired contrasts: the seeds are shared, so difference before resampling.
    reference = specs[0]["name"]
    for spec in specs[1:]:
        difference = per_seed[reference] - per_seed[spec["name"]]
        resampled = difference[draws].mean(axis=1)
        summary["contrasts"][f"{reference}_minus_{spec['name']}"] = {
            "mean": float(difference.mean()),
            "per_seed": difference.tolist(),
            "ci95": [float(np.percentile(resampled, low_q)),
                     float(np.percentile(resampled, high_q))],
        }

    return {"curves": curves, "summary": summary}


# =============================================================================
# Reporting
# =============================================================================

def print_summary(cfg: dict, result: dict) -> None:
    """Print mean success rate and paired contrasts, one row per method."""
    specs = parse_method_specs(cfg)
    summary = result["summary"]
    width = max(len(display_name(s)) for s in specs) + 2
    checkpoints = summary["checkpoints"]
    print()
    print("=" * 78)
    print(f"Mean success rate over checkpoints {checkpoints[0]}..{checkpoints[-1]} "
          f"({len(checkpoints)} points, {len(summary['seeds'])} seeds)")
    print("=" * 78)
    print(f"  {'Method':<{width}} {'Success':>10}  {'95% CI':>18}   {'vs Proposed':>22}")
    print(f"  {'-' * (width + 56)}")
    reference = specs[0]["name"]
    for spec in specs:
        entry = summary["methods"][spec["name"]]
        ci = f"[{entry['ci95'][0] * 100:5.2f}, {entry['ci95'][1] * 100:5.2f}]"
        key = f"{reference}_minus_{spec['name']}"
        if key in summary["contrasts"]:
            contrast = summary["contrasts"][key]
            delta = (f"{contrast['mean'] * 100:+6.2f} "
                     f"[{contrast['ci95'][0] * 100:+6.2f},"
                     f"{contrast['ci95'][1] * 100:+6.2f}]")
        else:
            delta = "—"
        print(f"  {entry['label']:<{width}} {entry['mean'] * 100:9.2f}%  "
              f"{ci:>18}   {delta:>22}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    """Run the requested stages and write everything to the run directory."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/fig4_vla.yaml")
    parser.add_argument("--stage", choices=["train", "score", "plot", "all"],
                        default="all")
    parser.add_argument("--seed", type=int, action="append",
                        help="restrict training to these seeds (repeatable)")
    parser.add_argument("--method", type=str, action="append",
                        help="restrict training to these methods (repeatable)")
    parser.add_argument("--save-adapters", action="store_true",
                        help="write trained adapters under the run directory")
    parser.add_argument("--verify-probes", action="store_true",
                        help="compare predictions to the archived probes")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="--verify-probes tolerance (default 1e-6, float32 epsilon)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg.get("device"))
    specs = parse_method_specs(cfg)
    if args.method:
        specs = [s for s in specs if s["name"] in args.method]
        if not specs:
            raise SystemExit(f"No configured method matches {args.method}")
    seeds = args.seed or cfg["seeds"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"out/vla_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("VLA online adaptation — four-method comparison")
    print("=" * 70)
    print(f"  Methods:    {', '.join(display_name(s) for s in specs)}")
    print(f"  Seeds:      {len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print(f"  Stream:     {cfg['train_pairs']} pairs, "
          f"{cfg['updates_per_arrival']} update each")
    print(f"  Frame:      {cfg['frame_angles_degrees']} deg, "
          f"every {cfg['regime_length']} pairs")
    print(f"  Device:     {device}")
    print(f"  Output:     {out_dir}/")

    probes = None
    if args.verify_probes:
        with open("data/vla/reference_probes.json") as f:
            probes = json.load(f)

    report = {}
    if args.stage in ("train", "all"):
        print()
        checkpoints = sorted(set(cfg["checkpoints"]) | {cfg["train_pairs"]})
        runs, started = [], time.time()
        for spec in specs:
            for seed in seeds:
                runs.append(train_seed(
                    spec, cfg, seed, device, checkpoints,
                    out_dir=out_dir if args.save_adapters else None,
                    probes=probes, tolerance=args.tolerance,
                ))
            recent = [r for r in runs if r["method"] == spec["name"]]
            line = (f"  {display_name(spec):<10} {len(recent):>3} seeds  "
                    f"{np.mean([r['ms_per_update_mean'] for r in recent]):6.3f} ms/update")
            if probes is not None:
                line += f"  max probe error {max(r['max_probe_error'] for r in recent):.2e}"
            print(line)
        print(f"  trained in {time.time() - started:.1f}s")
        report["training"] = runs

    if args.stage in ("score", "plot", "all"):
        result = score(cfg)
        report |= result
        print_summary(cfg, result)
        with open(os.path.join(out_dir, "curves.json"), "w") as f:
            json.dump(result["curves"], f, indent=2)
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(result["summary"], f, indent=2)

        if args.stage in ("plot", "all") and not args.no_plot:
            from utils.plot_vla_results import plot_curves
            written = plot_curves(cfg, result["curves"], os.path.join(out_dir, "plots"))
            print(f"\n  Plots saved to {out_dir}/plots/  ({', '.join(written)})")

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "device": str(device)}, f, sort_keys=False)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
