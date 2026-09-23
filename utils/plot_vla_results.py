"""
Figure for the VLA online adaptation experiment (paper Figure 4).

Success rate against training samples, one line per method, with a shaded
bootstrap interval and a dotted marker at each control-frame change.

Usage
-----
    from utils.plot_vla_results import plot_curves
    plot_curves(cfg, curves, "out/vla_.../plots")
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np


def plot_curves(cfg: dict, curves: list[dict], out_dir: str,
                shading: str = None) -> list[str]:
    """Render the figure as PDF and PNG.  Returns the filenames written."""
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    os.makedirs(out_dir, exist_ok=True)
    shading = shading or cfg.get("shading", "ci95")
    specs = cfg["methods"]
    lookup = {(row["method"], row["samples_seen"]): row for row in curves}

    points = cfg["checkpoints"]
    x = np.asarray(points)

    fig, ax = plt.subplots()
    for spec in specs:
        rows = [lookup[spec["name"], point] for point in points]
        line, = ax.plot(x, [r["mean"] for r in rows], label=spec["label"])
        if shading == "ci95":
            ax.fill_between(x, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows],
                            color=line.get_color(), alpha=0.2, linewidth=0)

    # The control frame yaws a further 18 degrees at each of these.
    changes = [c for c in range(cfg["regime_length"], cfg["train_pairs"],
                                cfg["regime_length"])]
    for change in changes:
        ax.axvline(change, color="grey", linestyle=":", linewidth=0.9)
    if changes:
        ax.plot([], [], color="grey", linestyle=":", linewidth=0.9,
                label="Change point")

    ax.set(xlim=(0, cfg["train_pairs"]), ylim=(0, 1),
           xlabel="Training samples", ylabel="Success rate")
    ax.legend()
    fig.tight_layout()

    written = []
    for extension in ("pdf", "png"):
        filename = f"vla_success_rate.{extension}"
        fig.savefig(os.path.join(out_dir, filename))
        written.append(filename)
    plt.close(fig)

    return written
