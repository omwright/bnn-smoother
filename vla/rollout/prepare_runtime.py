#!/usr/bin/env python3
"""
Check (or install) the external OpenPI/LIBERO runtime the rollout needs.

Run this before ``evaluate.py --execute``.  The failure it exists to catch is
the quiet one: an OpenPI checkout that looks right but is missing the compile
switch or the Transformers replacement files, which produces rollouts that run
and are wrong rather than rollouts that crash.

    uv run --project vla/rollout python vla/rollout/prepare_runtime.py \\
        --openpi-root ~/openpi --libero-root ~/LIBERO --check
    ... --apply     # install the patch and the replacement files

``--check`` is read-only and the default.  ``--apply`` writes into the OpenPI
checkout and into the installed Transformers package, so point it at a checkout
kept for this purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PATCHED_FILE = "src/openpi/models_pytorch/pi0_pytorch.py"
REPLACEMENT_PREFIX = "src/openpi/models_pytorch/transformers_replace/"


def require(condition, message):
    """Exit with ``message`` unless ``condition`` holds."""
    if not condition:
        raise SystemExit(f"error: {message}")


def sha(path: Path) -> str:
    """SHA-256 of a file, as hex."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_file(source: Path, destination: Path) -> None:
    """Replace one file atomically, leaving package-manager hardlinks intact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".runtime_", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        shutil.copymode(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    """Check the external checkouts, and install the patch with --apply."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs/fig4_vla.yaml")
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true",
                      help="install the OpenPI patch and the Transformers replacements")
    mode.add_argument("--check", action="store_true",
                      help="read-only verification (default)")
    args = parser.parse_args()

    require(sys.version_info[:2] == (3, 11),
            f"Use the pinned Python 3.11 rollout environment, not {sys.version.split()[0]}. "
            f"Run via `uv run --project vla/rollout`.")

    cfg = yaml.safe_load(args.config.read_text())
    spec = json.loads((HERE / "runtime_requirements.json").read_text())
    roots = {"openpi": args.openpi_root.expanduser().resolve(),
             "libero": args.libero_root.expanduser().resolve()}

    for name, root in roots.items():
        require(root.is_dir(), f"Missing {name} checkout: {root}")
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        expected = cfg["source_commits"][name]
        require(revision == expected,
                f"{name} is at {revision[:12]}, but the experiment recorded "
                f"{expected[:12]}.  Check out the recorded commit.")

    drift = []
    for name, expected in spec["dependencies"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            drift.append(f"{name}: not installed (recorded {expected})")
            continue
        if actual != expected:
            drift.append(f"{name}: {actual} installed, {expected} recorded")

    patch_needed, copies = False, []
    for relative, expected in spec["source_files"]["openpi"].items():
        source = roots["openpi"] / relative
        require(source.is_file(), f"Missing OpenPI source: {relative}")
        if relative == PATCHED_FILE and sha(source) != expected:
            original = subprocess.check_output(
                ["git", "-C", str(roots["openpi"]), "show", f"HEAD:{relative}"])
            require(source.read_bytes() == original,
                    f"{relative} has edits that are neither pristine nor the recorded "
                    f"patch; use a separate checkout kept for this experiment")
            patch_needed = True
        elif relative.startswith(REPLACEMENT_PREFIX):
            installed = (Path(importlib.metadata.distribution("transformers").locate_file(""))
                         / "transformers" / relative[len(REPLACEMENT_PREFIX):])
            if not installed.is_file() or sha(installed) != sha(source):
                copies.append((source, installed))

    changes = {"openpi_patch": patch_needed,
               "transformers_replacement_files": len(copies)}
    if not args.apply:
        require(not patch_needed and not copies,
                f"Runtime modifications are missing: {changes}.  Rerun with --apply.")
    else:
        if patch_needed:
            subprocess.run(["git", "-C", str(roots["openpi"]), "apply",
                            str(HERE / "openpi_runtime.patch")], check=True)
            require(sha(roots["openpi"] / PATCHED_FILE)
                    == spec["source_files"]["openpi"][PATCHED_FILE],
                    "Patched OpenPI source does not hash as recorded")
        for source, destination in copies:
            install_file(source, destination)

    print(json.dumps({
        "status": "passed",
        "mode": "apply" if args.apply else "check",
        "changes": changes,
        "dependency_drift": drift or "none — matches the recorded runtime",
    }, indent=2))


if __name__ == "__main__":
    main()
