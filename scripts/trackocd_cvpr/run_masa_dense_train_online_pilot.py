#!/usr/bin/env python3
"""Run the registered bounded MASA dense Train pilot in a fresh worker.

The existing frozen smoke executor supplies the model/input/resource guards.
This wrapper changes only the registered config/output namespace and never
changes the frozen model or the worker contract.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.trackocd_cvpr import audit_masa_dense_train_online_smoke as smoke

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/masa_dense_train_online_pilot_bdd32_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_pilot_bdd32_v1/result.json"
DEFAULT_RUN_ROOT = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_pilot_bdd32_v1/run"


def load_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "trackocd.cvpr.masa-dense-train-online-pilot.v1":
        raise ValueError("unexpected dense MASA pilot configuration")
    paths = value.get("image_paths")
    if not isinstance(paths, list) or len(paths) != 32 or len(set(paths)) != 32:
        raise ValueError("pilot requires exactly 32 distinct images")
    if paths != sorted(paths):
        raise ValueError("pilot image order must be deterministic")
    if value.get("source_split") != "train_image_only":
        raise ValueError("pilot is not Train-only")
    contract = value.get("causal_contract", {})
    if any(contract.get(key) is not expected for key, expected in {
        "future_observations_used": False,
        "gt_or_category_fields_read": False,
        "val_or_test_access": False,
        "training": False,
        "offline_filtering": False,
        "public_vocabulary_used": False,
        "formal_metrics": False,
    }.items()):
        raise ValueError("pilot causal boundary is not frozen")
    return value


def configure(config_path: Path, output: Path, run_root: Path) -> None:
    config = load_config(config_path)
    smoke.CONFIG = config_path
    smoke.OUTPUT = output
    smoke.RUN_ROOT = run_root
    smoke._config = lambda: config
    smoke._image_paths = lambda value: _image_paths(value)
    # The reused parent launches Path(__file__) from its module globals. Point
    # that child command back to this wrapper so it receives the same config.
    smoke.__file__ = str(Path(__file__).resolve())


def _image_paths(config: dict) -> list[Path]:
    """Resolve the registered BDD frame names while preserving the smoke guards."""
    result: list[Path] = []
    train_root = (smoke.FRAMES / "train").resolve()
    for raw in config["image_paths"]:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "train":
            raise ValueError("registered input is not Train-only")
        path = smoke.FRAMES / relative
        if not path.resolve().is_relative_to(train_root) or not path.is_file():
            raise ValueError("registered Train image is missing or escapes the Train tree")
        result.append(path)
    if len({path.parent for path in result}) != 1:
        raise ValueError("registered frames must come from one video directory")
    numbers = []
    for path in result:
        match = re.fullmatch(r"frame(\d+)", path.stem)
        if match is None:
            raise ValueError("registered BDD frames must use frameNNNN names")
        numbers.append(int(match.group(1)))
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError("registered frames are not consecutive")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    run_root = args.run_root.resolve()
    configure(config_path, output, run_root)
    if args.worker is not None:
        return smoke.worker(args.worker.resolve())
    return smoke.parent()


if __name__ == "__main__":
    raise SystemExit(main())
