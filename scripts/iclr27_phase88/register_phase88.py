#!/usr/bin/env python3
"""Register the isolated Phase88 window and immutable input lineage."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase88"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    started = dt.datetime.now(dt.timezone.utc)
    head = run("git", "rev-parse", "HEAD")
    remote = run("git", "ls-remote", "origin", "refs/heads/main").split()[0]
    assert head == remote, f"local HEAD {head} != origin/main {remote}"
    deadline = started + dt.timedelta(hours=10)
    input_paths = [
        ROOT / "AGENTS.md",
        ROOT / "research_log.md",
        ROOT / "docs/iclr27_phase87/PHASE87_AUTONOMOUS_RESEARCH_REPORT.md",
        ROOT / "outputs/iclr27_phase87/audit/final_decision.json",
        ROOT / "src/iclr27_phase19r/evaluation/internal.py",
        ROOT / "src/iclr27_phase19r/data/stream.py",
    ]
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    registration = {
        "schema_version": "trackocd.phase88.registration.v1",
        "phase": 88,
        "start_head": head,
        "origin_main_at_start": remote,
        "start_utc": started.isoformat(),
        "deadline_utc": deadline.isoformat(),
        "namespace": {
            "source": str((ROOT / "src/iclr27_phase88").resolve()),
            "scripts": str((ROOT / "scripts/iclr27_phase88").resolve()),
            "configs": str((ROOT / "configs/iclr27_phase88").resolve()),
            "docs": str((ROOT / "docs/iclr27_phase88").resolve()),
            "outputs": str((ROOT / "outputs/iclr27_phase88").resolve()),
            "large_storage": "/data2/usr_for_deadline/trackocd_phase88/project_outputs",
        },
        "frozen_prior_phase": 87,
        "frozen_phase87_head": "3c12af72a83681883cd6e4e5ec0beeb35d78bdd3",
        "protocol": {
            "train_only": True,
            "public_dev_q1_sealed_accessed": False,
            "future_rows_or_tracks": False,
            "ids_or_text_as_model_input": False,
            "phase19r_metric_source": "src/iclr27_phase19r/evaluation/internal.py",
            "formal_thresholds": {
                "commit_ct": 15,
                "category_coverage": 5,
                "video_coverage": 8,
                "existing_precision": 0.70,
                "negative_false_merge": 0.15,
                "known_micro": 0.206,
                "known_macro": 0.139,
            },
        },
        "gpu_preflight": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "reserved_external_devices_at_registration": [1, 2, 3, 4],
            "candidate_idle_devices_at_registration": [0, 5, 6, 7, 8, 9],
            "max_gpus": 4,
        },
        "input_hashes": input_hashes,
    }
    atomic_json(OUT / "audit/window_registration.json", registration)
    atomic_json(OUT / "audit/preregistration.json", {
        **registration,
        "hypothesis": "correct formal metrics plus real persistent memory, known branch, RESET targets, and visual-hard-negative source banks will improve full-runtime OCD without changing MOT or sealed protocol",
        "routes": ["C0V2", "C0_CONTINUE", "C1_SUPPORT", "ONE_TRAIN_EVIDENCE_REPAIR"],
        "stop_rules": ["data unavailable", "protocol impossible", "safety boundary", "unrecoverable resource", "same root cause across two authorized routes"],
        "not_stop_rules": ["scientific gate fail", "low precision", "false merge", "known low", "RESET low", "support no gain"],
    })
    print(json.dumps(registration, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
