#!/usr/bin/env python3
"""Register the independent Phase84 window and freeze its protocol metadata."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase84"
AUDIT = OUT / "audit"


def run(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return (p.stdout + p.stderr).strip()


def sha(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def iso(t: dt.datetime) -> str:
    return t.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    deadline = now + dt.timedelta(hours=10)
    phase83 = ROOT / "outputs/iclr27_phase83"
    prior = {
        "phase83_report": str((ROOT / "docs/iclr27_phase83/PHASE83_RESUMED_FINAL_REPORT.md").resolve()),
        "phase83_decision": str((ROOT / "outputs/iclr27_phase83/audit/resumed_phase83_decision.json").resolve()),
        "phase83_a2_native": str((Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")).resolve()),
        "phase83_a2_native_sha256": sha(Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")),
        "phase83_a2_dino": str((Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")).resolve()),
        "phase83_a2_dino_sha256": sha(Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")),
        "phase83_outputs_symlink": str(phase83),
        "phase83_outputs_target": os.readlink(phase83) if phase83.is_symlink() else None,
    }
    registration = {
        "schema_version": "trackocd.phase84.window_registration.v1",
        "phase": "Phase84",
        "status": "REGISTERED_STAGE0",
        "start_time_utc": iso(now),
        "deadline_utc": iso(deadline),
        "clock_restarted": True,
        "start_head": run("git", "rev-parse", "HEAD"),
        "git_status": run("git", "status", "--short"),
        "origin_main": run("git", "ls-remote", "origin", "refs/heads/main"),
        "cwd": str(ROOT),
        "resources": {
            "date_utc": run("date", "-u"),
            "nvidia_smi": run("nvidia-smi"),
            "free_h": run("free", "-h"),
            "disk_data1_data2": run("df", "-h", "/data1", "/data2"),
            "process_count": run("bash", "-lc", "ps -e --no-headers | wc -l"),
        },
        "protocol": {
            "r_queries": 984,
            "o_positive_events": 76,
            "o_negative_events": 76,
            "prefixes": [1, 2, 4, 8, 16],
            "reliable_rule": "assigned == 1 and transformed_or_row_iou >= 0.5; historical frozen O remains posthoc comparator",
            "candidate_universe": "Phase75D exact all validation keys except self and same video; native Q0 for Phase84 support",
            "physical_feature_source": "frozen Phase75D aligned DINOv2 (0.8 CLS + 0.2 ROI as recorded), unchanged for R attribution",
            "physical_reassociation": "Phase82R temporal appearance mean, dormant-only, observed-step causal timing, gap<=16, accept score=0.5, same-frame collision safety",
            "source_conditioned_support": "same corrected native DINOv2 source/target, source mean plus fixed M=3 causal prototypes, candidate-set softmax with explicit DEFER",
        },
        "boundaries": {
            "public_dev_q1_sealed_accessed": False,
            "future_rows_or_tracks": False,
            "category_text_as_input": False,
            "semantic_id_as_input": False,
            "physical_id_as_feature": False,
            "event_labels_as_model_input": False,
            "train_gt_only_for_supervision": True,
            "old_phase83_artifacts_modified": False,
            "controller_run": False,
            "sealed_run": False,
            "threshold_sweep": False,
        },
        "branches": {
            "A84P": "true full physical reassociation -> frozen raw R",
            "B84S": "source-conditioned same-space native candidate matcher with DEFER",
            "B84A": "alignment only if B84S selection is materially improved",
            "C84": "unchanged controller only after safe P/O->R improvement",
        },
        "prior_phase83_reclassification": {
            "A2_report_source_bug": True,
            "A2_membership_not_reassociated": True,
            "B2_B3_B4_query_agnostic": True,
            "B5_mixed_feature_space": True,
            "interpretation": "Phase83 localized interface errors; it did not exhaust the physical-to-R or source-conditioned support hypotheses.",
        },
        "prior_artifacts": prior,
    }
    atomic_json(AUDIT / "window_registration.json", registration)
    atomic_json(AUDIT / "finalization_lock.json", {"allowed": False, "deadline_utc": iso(deadline), "reason": "Phase84 window open; final report prohibited before deadline-minus-45-minutes", "updated_utc": iso(now)})
    atomic_json(OUT / "status.json", {"phase": "Phase84", "status": "REGISTERED_STAGE0", "start_time_utc": iso(now), "deadline_utc": iso(deadline), "start_head": registration["start_head"], "next_action": "Phase83 integrity audit and physical/support interface contracts", "public_dev_q1_sealed_accessed": False, "controller_run": False, "sealed_run": False})
    print(json.dumps({"status": "REGISTERED_STAGE0", "start_time_utc": iso(now), "deadline_utc": iso(deadline), "start_head": registration["start_head"]}, indent=2))


if __name__ == "__main__":
    main()
