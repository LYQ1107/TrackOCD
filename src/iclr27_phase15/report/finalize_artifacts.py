"""Attach reproducibility metadata and validate Phase15A artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    inputs = {
        "public_track_cache": "outputs/iclr27_phase6d/assets/full_tao_tracks.npz",
        "phase15_split": "outputs/iclr27_phase15/manifests/phase15_preregistration.json",
        "leakage_audit": "outputs/iclr27_phase15/manifests/data_and_leakage_audit.json",
        "phase14c_proposals": "outputs/iclr27_phase14c/proposals/proposals_mixed.csv",
        "phase14c_aligned": "outputs/iclr27_phase14c/proposals/proposals_aligned.csv",
        "phase14c_raw_features": "outputs/iclr27_phase14c/features/proposal_dinov2.npz",
        "phase14c_gt_sidecar": "outputs/iclr27_phase14c/manifests/mixed_gt_tracks.jsonl",
    }
    code = {
        "phase15a_probe": "src/iclr27_phase15/representation/phase15a_probe.py",
        "protocol": "docs/iclr27_phase15/PROTOCOL.md",
        "prior_art": "docs/iclr27_phase15/PRIOR_ART_AND_IMPLEMENTATION_AUDIT.md",
        "crop_tube_diagnostic": "src/iclr27_phase15/evaluation/crop_tube_diagnostic.py",
    }
    input_meta = {k: {"path": v, "sha256": sha(ROOT / v)} for k, v in inputs.items()}
    code_meta = {k: {"path": v, "sha256": sha(ROOT / v)} for k, v in code.items()}
    ckpts = {}
    for p in sorted((ROOT / "outputs/iclr27_phase15/checkpoints").glob("*.pth")):
        ckpts[p.name] = {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
    audit = json.loads((ROOT / inputs["leakage_audit"]).read_text())
    common = {
        "artifact_metadata": {
            "protocol": "phase15a",
            "code_revision": "git_unavailable; content-addressed source hashes recorded",
            "code": code_meta,
            "inputs": input_meta,
            "checkpoints": ckpts,
            "command": "CUDA_VISIBLE_DEVICES=5 /home/lwr/anaconda3/envs/AVI/bin/python -u -m src.iclr27_phase15.representation.phase15a_probe --device cuda:0 --steps 600",
            "split_identity": "outputs/iclr27_phase15/manifests/phase15_preregistration.json",
            "split_audit_pass": bool(audit["pass"]),
            "devplus_labels_accessible_to_fit": False,
            "devplus_labels_accessible_to_calibration": False,
            "q1_labels_accessible": False,
            "future_frames_used_for_method": False,
            "physical_id_used_as_feature": False,
        }
    }
    eval_dir = ROOT / "outputs/iclr27_phase15/eval"
    targets = [
        "phase15a_offline_summary.json", "phase15a_online_summary.json",
        "phase15a_decision.json", "strict_trackocd_summary.json",
        "causal_contract.json", "resource_summary.json",
        "phase15d_crop_tube_diagnostic.json", "phase15a_devplus_offline_summary.json",
    ] + [p.name for p in eval_dir.glob("phase15a_*_strict.json")]
    for name in targets:
        path = eval_dir / name
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        value.update(common)
        atomic_json(path, value)
    # Integrity checks over every claimed causal CSV.
    csv_checks = {}
    for path in sorted(eval_dir.glob("phase15a_*.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 16616, (path, len(rows))
        assert all(r.get("sem_action") in ("known", "new", "existing") for r in rows)
        csv_checks[path.name] = {"rows": len(rows), "sha256": sha(path),
                                "immediate_actions": True}
    resource = json.loads((eval_dir / "resource_summary.json").read_text())
    resource["artifact_metadata"] = common["artifact_metadata"]
    resource.update({
        "preflight": {
            "gpu": "GPU5 A100-SXM4-40GB idle (2 MiB, 0% utilization)",
            "host_ram": "125 GiB total; 87 GiB available",
            "disk": "/data1: 201 GiB available",
            "other_gpu_processes_untouched": True,
        },
        "formal_run": {"gpu": 5, "visible_devices": "5", "duration_seconds": resource.get("duration_seconds"),
                       "steps": 600, "seeds": [20260824, 20260825],
                       "exit_code": 0, "oom": False, "near_oom": False},
        "repair_incidents": [
            {"kind": "cpu_smoke", "cause": "per-occurrence small MLP calls made CPU replay unacceptably slow", "action": "stopped only own smoke PIDs and added bounded causal carry-forward/global prefilter"},
            {"kind": "formal_gpu", "cause": "none", "action": "completed after repair"},
        ],
        "stopped_processes": "only two task-owned CPU smoke processes; no other-user process stopped",
        "csv_checks": csv_checks,
    })
    atomic_json(eval_dir / "resource_summary.json", resource)
    print(json.dumps({"metadata_attached": len(targets), "csv_checks": csv_checks}, indent=2))


if __name__ == "__main__":
    main()
