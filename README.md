# bnn-smoothing

This code accompanies

> O. Wright, H. Jing, Q. Shen, K. Niinuma, Y. Nakahira and J. M. F. Moura,
> "Full-Covariance Smoothing of Bayesian Neural Networks for Online
> Adaptation," *IEEE Conference on Decision and Control (CDC)*, 2026.

The proposed method trains Bayesian neural networks via closed-form moment propagation and Rauch–Tung–Striebel smoothing, in one pass over the data.
Each arriving sample is propagated forward through the network, and a backward smoothing pass updates the weight posterior in closed form.

## Setup

We use [uv](https://docs.astral.sh/uv/) for managing this project. With that installed, run

```bash
uv sync --extra baselines
```

to recreate the experiments.

## Reproducing the figures

| Paper item | Command |
|---|---|
| Figure 1 — rotating moons | `uv run python run_online_moons_experiment.py --config configs/fig1_moons.yaml` |
| Figure 2, Table I — CartPole | `uv run python run_dynamics_experiment.py --config configs/fig2_cartpole.yaml` |
| Figure 3, Table II — Industrial Benchmark | `uv run python run_dynamics_experiment.py --config configs/fig3_industrial.yaml` |
| Figure 4 — VLA online adaptation | `uv run python run_vla_experiment.py --config configs/fig4_vla.yaml` |

Each run writes a timestamped directory under `out/`, which is git-ignored:

- `aggregated.json` — the curves behind the figures, averaged across trials
- `plots/` — the figures, under matplotlib's default style (PNG, and PDF for Figure 4)
- `config.yaml` — the exact configuration the run used
- `traces.json`, `rollout.json` — per-trial detail, where applicable

Add `--no-plot` to skip the figures, or `--no-rollout` (dynamics only) to skip
the multi-step rollout evaluation.

## Note on the VLA experiment

**The 27,000 episode outcomes are committed**, in `data/vla/episodes.csv.gz`.
Producing them needs the pi0.5 weights, LIBERO and a GPU, so they are archived
measurements.  Training retrains all 60 adapters (four methods × 15 seeds) from the committed action streams in about 8 seconds.

```bash
# Stages: train, score, plot, or all (the default)
uv run python run_vla_experiment.py --config configs/fig4_vla.yaml --stage train

# Check retrained adapters against the archived predictions
uv run python run_vla_experiment.py --config configs/fig4_vla.yaml --verify-probes

# Keep the trained adapters, for fresh rollouts
uv run python run_vla_experiment.py --config configs/fig4_vla.yaml --save-adapters
```

`--verify-probes` is a check: if you change a prior it is *meant* to fail.

### Fresh rollouts need a second environment

Re-running the policy in LIBERO needs a CUDA GPU, the pi0.5 weights, checkouts of OpenPI and LIBERO, and unlike the rest of the experiments here this will take many GPU-hours to run. It needs its own environment using `torch==2.7.1` and `numpy==1.26.4`, as opposed to our `torch>=2.8` and `numpy>=2.3`.  So it is its own uv project, with its own lock file:

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
every command runs from the repository root and `configs/…` and `out/…` mean the
same thing in each. OpenPI and LIBERO install
from pinned git checkouts; the pi0.5 weights are never bundled or downloaded.

## Repository layout

- `bnn/` — The proposed method: moment propagation, covariance storage, the smoother
- `methods/` — Every method behind one interface, selected by name in a config
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
