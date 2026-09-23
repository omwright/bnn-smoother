#!/usr/bin/env python3
"""
LIBERO rollout evaluation for the VLA adaptation experiment.

Loads a trained adapter, runs it on top of the frozen pi0.5 policy in LIBERO,
and records one JSON per episode.  Dry-run by default: ``--execute`` is what
loads the GPU policy and actually simulates.

Runs in the pinned rollout environment, not the repo's:

    uv sync --project vla/rollout                      # once
    uv run --project vla/rollout python vla/rollout/evaluate.py --help

``--project`` swaps the virtual environment and leaves the working directory
alone, so run this from the repo root and every relative path means what it
usually means.

Adapters come from ``run_vla_experiment.py --save-adapters``.  An adapter that
differs from the archived run is accepted: changing a prior and re-running is
the point of having this here.  What is still checked is that the adapter's
identity and normalisation match the stream it claims, which catches the
mistakes that actually happen.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def require(condition, message):
    """Exit with ``message`` unless ``condition`` holds."""
    if not condition:
        raise SystemExit(f"error: {message}")


def write(path: Path, value) -> None:
    """Write ``value`` as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


# =============================================================================
# Selection
# =============================================================================

def parse_args(argv=None):
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs/fig4_vla.yaml")
    parser.add_argument("--adapters", type=Path, required=True,
                        help="directory written by run_vla_experiment.py --save-adapters")
    parser.add_argument("--output", type=Path, required=True,
                        help="results directory, outside the adapter directory")
    for name in ("openpi-root", "libero-root", "base-checkpoint"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path,
                        help="LIBERO assets (default: LIBERO_ROOT/libero/libero/assets)")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--method", type=str, action="append")
    parser.add_argument("--samples-seen", type=int, action="append")
    parser.add_argument("--task", type=int, action="append")
    parser.add_argument("--episode", type=int, action="append",
                        help="index 0..4, selecting initial states 10..14")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split evaluation points across parallel workers")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="load the GPU policy and run the plan")
    mode.add_argument("--dry-run", action="store_true", help="print the plan (default)")
    return parser.parse_args(argv)


def selection(args, cfg):
    """Expand the CLI selectors into an ordered list of episode cells."""
    require(args.num_shards > 0 and 0 <= args.shard_index < args.num_shards,
            f"Invalid shard {args.shard_index}/{args.num_shards}")

    def pick(chosen, options, label):
        """Keep ``options`` in order, restricted to ``chosen`` if given."""
        if not chosen:
            return list(options)
        unknown = [c for c in chosen if c not in options]
        require(not unknown, f"{label} {unknown} absent from the configuration")
        return [c for c in options if c in chosen]

    names = [spec["name"] for spec in cfg["methods"]]
    points = pick(args.samples_seen, cfg["checkpoints"], "checkpoint")
    points = points[args.shard_index::args.num_shards]
    require(points, "No evaluation points assigned to this shard")
    return list(itertools.product(
        pick(args.seed, cfg["seeds"], "seed"),
        pick(args.method, names, "method"),
        points,
        pick(args.task, cfg["tasks"], "task"),
        pick(args.episode, list(range(cfg["episodes_per_task"])), "episode"),
    ))


# =============================================================================
# Adapter restoration
# =============================================================================

def restore_adapter(path: Path, name: str, seed: int, step: int, cfg: dict):
    """Rebuild a saved adapter and return ``(predict, normalization)``.

    ``predict`` maps normalised actions to physical-unit residuals; that plain
    callable is the entire interface the rollout loop needs, which is why a new
    method costs nothing here.
    """
    import numpy as np
    import torch
    from data.vla_streams import load as load_stream
    from run_vla_experiment import build_model, parse_method_specs

    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = next((s for s in parse_method_specs(cfg) if s["name"] == name), None)
    require(spec is not None, f"Method {name!r} is not in {cfg}")
    for key, expected in (("seed", seed), ("samples_seen", step)):
        require(payload.get(key) == expected,
                f"{path}: adapter {key} is {payload.get(key)!r}, expected {expected!r}")

    method = build_model(spec, cfg, seed, torch.device("cpu"))
    _load_state(name, method.net, payload)

    stream = load_stream(seed)
    normalization = {}
    for field, expected in (("ref_mean", stream.ref_mean), ("ref_std", stream.ref_std)):
        stored = payload[field].double()
        require(torch.equal(stored, torch.tensor(expected, dtype=torch.float64)),
                f"{path}: {field} differs from seed {seed}'s stream")
        normalization[field] = stored
    require(bool(torch.isfinite(normalization["ref_mean"]).all())
            and bool((normalization["ref_std"] > 0).all()),
            f"{path}: invalid normalization")

    @torch.no_grad()
    def predict(x):
        """Residual for one action or a batch of them."""
        x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
        single = x.ndim == 1
        out = method.predict(x.unsqueeze(0) if single else x)[0].double()
        return out.squeeze(0) if single else out

    return predict, normalization


def _load_state(name: str, net, payload: dict) -> None:
    """Write a saved state back into a freshly built net."""
    if name == "kbnn_onestep":
        for layer, mean, cov in zip(net.layers, payload["weight_means"], payload["weight_covs"]):
            layer.weight_mean = mean
            layer._cov.set_covariance(cov)
    elif name == "wagner_kbnn":
        net.mw, net.Cw = payload["mw"], payload["Cw"]
    elif name == "tagi":
        for field in ("mu_w", "var_w", "mu_b", "var_b"):
            setattr(net, field, payload[field])
    elif name == "replay_nn":
        net.load_training_state(payload)
        net.eval()
    else:
        raise ValueError(f"No state layout for method {name!r}")


# =============================================================================
# One episode
# =============================================================================

def episode(policy, env, state, prompt, seed, predict, normalization, theta, cfg):
    """Run one LIBERO episode with the adapter in the loop.

    Preprocessing, seeding, residual application and the action budget are the
    archived ones.  The frame change is emulated by rotating the *executed*
    action rather than moving the camera, so a perfect residual cancels the
    rotation exactly and recovers the nominal command.
    """
    import numpy as np
    import torch
    from data.vla_streams import rotation
    from openpi_client import image_tools

    from libero_helpers import LIBERO_DUMMY_ACTION, _quat2axisangle

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    env.reset()
    observation = env.set_init_state(state)

    queue = collections.deque()
    inverse = torch.tensor(rotation(-theta), dtype=torch.float64)
    settle, horizon = cfg["settle_steps"], cfg["maximum_action_steps"]
    size = cfg["policy_image_size"]

    for t in range(settle + horizon):
        if t < settle:
            observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        if not queue:
            def image(name):
                """One camera view, flipped and resized for the policy."""
                flipped = np.ascontiguousarray(observation[name][::-1, ::-1])
                return image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(flipped, size, size))

            element = {
                "observation/image": image("agentview_image"),
                "observation/wrist_image": image("robot0_eye_in_hand_image"),
                "observation/state": np.concatenate((
                    observation["robot0_eef_pos"],
                    _quat2axisangle(observation["robot0_eef_quat"]),
                    observation["robot0_gripper_qpos"])),
                "prompt": str(prompt),
            }
            actions = np.asarray(policy.infer(element)["actions"]).copy()
            action = torch.tensor(actions[:, :7], dtype=torch.float64)
            xs = (action - normalization["ref_mean"]) / normalization["ref_std"]
            action = action + torch.stack([predict(x) for x in xs])
            action = action @ inverse.T
            require(bool(torch.isfinite(action).all()), "Nonfinite action")
            actions[:, :7] = action.numpy()
            queue.extend(actions[:cfg["replan_steps"]])
        observation, _, done, _ = env.step(queue.popleft().tolist())
        if done:
            return True, t - settle + 1
    return False, horizon


# =============================================================================
# Execution
# =============================================================================

def runtime_setup(args, cfg):
    """Put OpenPI and LIBERO on the path and point LIBERO at its assets."""
    openpi = args.openpi_root.resolve()
    libero = args.libero_root.resolve()
    benchmark = libero / "libero/libero"
    assets = (args.assets_dir or benchmark / "assets").resolve()

    paths = [Path(__file__).resolve().parent, openpi / "src",
             openpi / "packages/openpi-client/src", openpi, libero]
    for path in paths:
        require(path.is_dir(), f"Missing runtime source directory: {path}")
    for path in (benchmark / "bddl_files", benchmark / "init_files", assets):
        require(path.is_dir(), f"Missing external LIBERO data directory: {path}")
    sys.path[:0] = [str(p) for p in paths]

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    configuration = args.output / "libero_config"
    configuration.mkdir(parents=True, exist_ok=True)
    content = {"benchmark_root": benchmark, "bddl_files": benchmark / "bddl_files",
               "init_states": benchmark / "init_files",
               "datasets": libero / "datasets", "assets": assets}
    (configuration / "config.yaml").write_text(
        "".join(f"{k}: {json.dumps(str(v))}\n" for k, v in content.items()))
    os.environ["LIBERO_CONFIG_PATH"] = str(configuration)
    return openpi, libero


def execute(args, cfg, selected):
    """Run every selected episode in the simulator, writing one JSON each."""
    import numpy as np
    import torch

    runtime_setup(args, cfg)
    from libero.libero import benchmark, get_libero_path
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    from libero_helpers import _get_libero_env

    require(torch.cuda.is_available(), "A CUDA GPU is required for the pi0.5 rollout")
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    print("Loading pi0.5 policy on CUDA", flush=True)
    policy = policy_config.create_trained_policy(
        openpi_config.get_config("pi05_libero"), args.base_checkpoint,
        pytorch_device="cuda")
    suite = benchmark.get_benchmark_dict()[cfg["suite"]]()
    regime = cfg["regime_length"]
    angles = np.deg2rad(cfg["frame_angles_degrees"])

    rows = []
    for (seed, name, step), group in itertools.groupby(selected, key=lambda c: c[:3]):
        adapter = args.adapters / str(seed) / name / f"step_{step:03d}.pt"
        require(adapter.is_file(), f"Missing adapter {adapter}; run --save-adapters first")
        predict, normalization = restore_adapter(adapter, name, seed, step, cfg)
        theta = float(angles[min(step // regime, len(angles) - 1)])

        for task_id, task_group in itertools.groupby(group, key=lambda c: c[3]):
            task = suite.get_task(task_id)
            states_path = (Path(get_libero_path("init_states"))
                           / task.problem_folder / task.init_states_file)
            states = torch.load(states_path, map_location="cpu", weights_only=False)
            env = None
            try:
                for _, _, _, _, ep in task_group:
                    path = args.output / f"episodes/{seed}/{name}/{step:03d}/{task_id}_{ep}.json"
                    if path.exists():
                        rows.append(json.loads(path.read_text()))
                        continue
                    if env is None:
                        env, prompt = _get_libero_env(
                            task, cfg["image_size"], seed + 1000, 0.0, 0.0, None, 1.0)
                    started = time.monotonic()
                    success, steps = episode(
                        policy, env, states[cfg["initial_state_indices"][ep]], prompt,
                        seed + 1000 + ep, predict, normalization, theta, cfg)
                    row = {"seed": seed, "method": name, "samples_seen": step,
                           "task": task_id, "episode": ep,
                           "initial_state_index": cfg["initial_state_indices"][ep],
                           "success": bool(success), "steps": steps, "theta": theta,
                           "seconds": time.monotonic() - started, "status": "complete"}
                    write(path, row)
                    rows.append(row)
                    print(seed, name, step, task_id, ep, "success", success,
                          "steps", steps, flush=True)
            finally:
                if env is not None:
                    env.close()

    write(args.output / "results.json",
          {"status": "complete", "episodes": len(rows),
           "successes": sum(r["success"] for r in rows), "rows": rows})


def main(argv=None):
    """Select the episodes, then dry-run or execute them."""
    import yaml

    args = parse_args(argv)
    for name in ("adapters", "output", "openpi_root", "libero_root", "base_checkpoint"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    require(args.output != args.adapters, "Keep rollout output separate from adapters")

    cfg = yaml.safe_load(args.config.read_text())
    selected = selection(args, cfg)
    plan = {"mode": "execute" if args.execute else "dry-run", "episodes": len(selected),
            "seeds": sorted({c[0] for c in selected}),
            "methods": sorted({c[1] for c in selected}),
            "checkpoints": sorted({c[2] for c in selected}),
            "shard": f"{args.shard_index}/{args.num_shards}",
            "adapters": str(args.adapters), "output": str(args.output)}
    print(json.dumps(plan, indent=2), flush=True)

    if args.execute:
        execute(args, cfg, selected)


if __name__ == "__main__":
    main()
