# bnn-smoother

This code accompanies

> O. Wright, H. Jing, Q. Shen, K. Niinuma, Y. Nakahira and J. M. F. Moura,
> "[Full-Covariance Smoothing of Bayesian Neural Networks for Online
> Adaptation](https://arxiv.org/abs/2609.27244v1)," *IEEE Conference on Decision and Control (CDC)*, 2026.

The proposed method trains Bayesian neural networks via closed-form moment propagation and Rauch–Tung–Striebel smoothing, in one pass over the data.
Each arriving sample is propagated forward through the network, and a backward smoothing pass updates the weight posterior in closed form.

## Install

We use [uv](https://docs.astral.sh/uv/) for managing this project. With uv installed, run

```bash
uv sync --extra baselines
```

to install dependencies.

## Run

| Paper item | Command |
|---|---|
| Figure 1 — Rotating moons | `uv run python run_online_moons_experiment.py --config configs/fig1_moons.yaml` |
| Figure 2, Table I — CartPole | `uv run python run_dynamics_experiment.py --config configs/fig2_cartpole.yaml` |
| Figure 3, Table II — Industrial Benchmark | `uv run python run_dynamics_experiment.py --config configs/fig3_industrial.yaml` |
| Figure 4 — VLA online adaptation | `uv run python run_vla_experiment.py --config configs/fig4_vla.yaml` |

Each run writes a timestamped directory under `out/`:

- `aggregated.json` — aggregate data
- `plots/` — matplotlib figures
- `config.yaml` — configuration used
- `traces.json`, `rollout.json` — per-trial details where applicable

Add `--no-plot` to skip the figures, or `--no-rollout` (dynamics only) to skip
the multi-step rollout evaluation.

## Note on the VLA experiment

The episode outcomes are archived in `data/vla/episodes.csv.gz`, but they can be reproduced with the pi0.5 weights, LIBERO, and a GPU.

```bash
# Stages: train, score, plot, or all (the default)
uv run python run_vla_experiment.py --config configs/fig4_vla.yaml --stage train

# Check retrained adapters against the archived predictions
uv run python run_vla_experiment.py --config configs/fig4_vla.yaml --verify-probes

# Keep the trained adapters, for fresh rollouts
uv run python run_vla_experiment.py --config configs/fig4_vla.yaml --save-adapters
```

(`--verify-probes` is a check: if you change config/inputs it is meant to fail.)

### Fresh rollouts need a second environment

Re-running the policy in LIBERO needs a CUDA GPU, the pi0.5 weights, checkouts of OpenPI and LIBERO, and unlike the rest of the experiments here this will take many GPU-hours to run. Due to dependency issues (it needs `torch==2.7.1` and `numpy==1.26.4` as opposed to `torch>=2.8` and `numpy>=2.3`), it is its own uv project:

```bash
# One-time: build vla/rollout/.venv at the recorded pins
uv sync --project vla/rollout
```

```bash
# Check the external checkouts and install the required OpenPI patch
uv run --project vla/rollout python vla/rollout/prepare_runtime.py \
    --openpi-root ~/openpi --libero-root ~/LIBERO --apply
```

```bash
# Evaluate saved adapters in the simulator
uv run --project vla/rollout python vla/rollout/evaluate.py \
    --config configs/fig4_vla.yaml \
    --adapters out/vla_<timestamp>/adapters \
    --output out/vla_rollout_<timestamp> \
    --openpi-root ~/openpi --libero-root ~/LIBERO \
    --base-checkpoint ~/checkpoints/pi05_libero_pytorch
```

`--project` selects the environment and leaves your working directory alone, so
every command runs from the repository root. OpenPI and LIBERO install
from pinned git checkouts.

## Repository layout

- `bnn/` — The proposed smoother, moment propagation, covariance storage
- `methods/` — Interface for all methods under consideration
- `data/` — Stream and environment generators, plus the archived VLA data
- `configs/` — Config files
- `utils/` — Helper functions
- `vla/rollout/` — Separate uv project for LIBERO simulator rollouts
- `out/` — Experiment outputs (git-ignored)

## Cite

```bibtex
@inproceedings{wright2026fullcovariance,
  author    = {Wright, Oren and Jing, Haoming and Shen, Qiaoan and
               Niinuma, Koichiro and Nakahira, Yorie and Moura, Jos\'{e} M. F.},
  title     = {Full-Covariance Smoothing of {B}ayesian Neural Networks for
               Online Adaptation},
  booktitle = {IEEE Conference on Decision and Control (CDC)},
  year      = {2026},
}
```

## License

MIT license except where noted: `vla/rollout/libero_helpers.py` and
`vla/rollout/openpi_runtime.patch` derive from LIBERO and OpenPI, whose
licenses are in `vla/rollout/licenses/`.
