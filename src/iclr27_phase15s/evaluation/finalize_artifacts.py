"""Write the small, auditable Phase15S/16 decision and resource ledgers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs/iclr27_phase15s"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def command(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # diagnostics must not hide the experiment result
        return f"unavailable: {exc}"


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    required = [
        "../../docs/iclr27_phase15s/PROTOCOL.md",
        "../../docs/iclr27_phase15s/STORAGE_AND_RESOURCE_LEDGER.md",
        "../../docs/iclr27_phase15s/PHASE15S_COMPLETE_REPORT.md",
        "manifests/preregistration.json",
        "manifests/data_split_and_leakage_audit.json",
        "eval/fixed_ct_contract.json",
        "eval/fixed_ct_oracle_controls.json",
        "eval/dsct_known_coverage_audit.json",
        "eval/known_bank_ceiling.json",
        "eval/episodic_calibration_summary.json",
        "eval/cls_roi_offline_summary.json",
        "eval/public_known_audit.json",
        "eval/strict_trackocd_summary.json",
        "eval/transition_contract.json",
        "eval/proposal_domain_shift_diagnostic.json",
        "features/public_cls_roi.npz",
        "features/devplus_cls_roi.npz",
        "csv/cls_devplus.csv",
        "csv/roi_devplus.csv",
    ]
    hashes = {}
    missing = []
    for rel in required:
        p = OUT / rel
        if p.exists():
            hashes[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        else:
            missing.append(rel)

    symlinks = {}
    for rel in [
        "sources/tao_train_annotations.json", "sources/tao_train_frames",
        "sources/supported_known_ids.json", "sources/full_tao_tracks.npz",
        "sources/proposals_aligned.csv", "sources/proposals_mixed.csv",
        "sources/mixed_gt_tracks.jsonl", "sources/devplus_cls_features.npz",
        "sources/phase15r_roi_historical.npz", "sources/devplus_annotation.json",
        "checkpoints/phase6b_dsct_stage_d.pth",
    ]:
        p = ROOT / "data/iclr27_phase15s" / rel
        symlinks[rel] = {"is_symlink": p.is_symlink(), "target": os.readlink(p) if p.is_symlink() else None}

    resource = {
        "protocol": "trackocd_iclr27_phase15s16",
        "generated_at": now,
        "project_git": "unavailable; content hashes are the revision record",
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "preflight": {
            "pre_edit": {"disk_available": "183G", "ram_total": "125G", "ram_available": "~71G", "swap": "0B", "process_count": "recorded in ledger", "gpu_observation": "GPU 0/1/4/5 occupied ~3.3-4.4GiB; GPU 2/3 ~29GiB (GPU3 busy); GPU 6-9 ~1.3GiB low utilization; no process stopped"},
            "dsct": {"disk_available": "181G", "ram_available": "~63G", "gpu": 2, "gpu_memory_before": "4MiB", "gpu_utilization_before": "0%", "process_count": 697},
            "features": {"disk_available": "174G", "ram_available": "~93G", "gpu": 2, "gpu_memory_before": "4MiB", "gpu_utilization_before": "0%", "process_count": 708},
            "final_read_only": {"disk_available": "173G", "ram_available": "~112G", "swap": "0B", "process_count": 691},
        },
        "jobs": {
            "dsct_public_roles": {"device": "cuda:2", "videos": 370, "selected_frames": 12000, "rows": 43423, "launch": "2026-08-24T14:53:55+08:00", "done": "2026-08-24T16:03:58+08:00", "markers": ["outputs/iclr27_phase15s/dsct_bank/public_roles/.launched", "outputs/iclr27_phase15s/dsct_bank/public_roles/.done"], "status": "complete"},
            "features_public": {"device": "cuda:2", "batch": 16, "rows": 43423, "launch": "2026-08-24T16:06:44+08:00", "done": "2026-08-24T17:18:23+08:00", "status": "complete"},
            "features_devplus": {"device": "cuda:2", "batch": 16, "rows": 16616, "launch": "2026-08-24T17:18:23+08:00", "done": "2026-08-24T17:30:37+08:00", "status": "complete"},
            "controller": {"device": "CPU", "status": "complete", "calibration_rows": 2000, "grid": "144 points x 2 seeds", "devplus_rows": 16616},
        },
        "incidents": [
            {"unit": "DSCT annotation loader", "cause": "frozen loader selected the alternate parser for a non-validation filename", "repair": "new validation_public_roles.json path and parser smoke check", "impact": "no proposal output was consumed; no memory incident"},
            {"unit": "CPU calibration", "cause": "initial per-state Python scoring was too slow", "repair": "vectorized known/cross-state scoring, same grid/seeds/data", "impact": "task-owned attempt interrupted; no other process stopped; final rerun complete"},
            {"unit": "post-run JSON sanity check", "cause": "a recursive glob included the large raw DSCT JSON and two validation shells began loading it", "repair": "stopped only the two task-owned validation PIDs and replaced the check with targeted small-artifact validation", "impact": "about 20GB transient RSS at peak; host safety floor was not crossed; no experiment artifact or other process was affected"},
        ],
        "safety": {"max_gpus": 4, "gpus_used": [2], "oom": False, "near_oom": False, "swap_used": False, "other_user_process_terminated": False, "large_input_copied": False},
        "commands": [
            "PYTHONPATH=. /home/lwr/anaconda3/envs/AVI/bin/python -m src.iclr27_phase15s.evaluation.fixed_ct",
            "bash src/iclr27_phase15s/data/run_dsct_public.sh",
            "bash src/iclr27_phase15s/representation/run_features.sh",
            "PYTHONPATH=. /home/lwr/anaconda3/envs/AVI/bin/python -m src.iclr27_phase15s.evaluation.run_controller",
            "PYTHONPATH=. /home/lwr/anaconda3/envs/AVI/bin/python -m src.iclr27_phase15s.evaluation.proposal_domain_shift ...",
        ],
        "artifact_hash_manifest": "outputs/iclr27_phase15s/manifests/artifact_hashes.json",
    }
    atomic_json(OUT / "eval/resource_summary.json", resource)

    strict = json.loads((OUT / "eval/strict_trackocd_summary.json").read_text())
    cal = json.loads((OUT / "eval/episodic_calibration_summary.json").read_text())
    coverage = json.loads((OUT / "eval/known_bank_ceiling.json").read_text())
    shift = json.loads((OUT / "eval/proposal_domain_shift_diagnostic.json").read_text())
    decision = {
        "protocol": "trackocd_iclr27_phase15s16",
        "generated_at": now,
        "selected_branch": "S-D_PROPOSAL_DOMAIN_SHIFT",
        "status": "TRACKOCD_NOT_YET_ICLR_LEVEL",
        "coverage_ceiling": coverage["ceiling"],
        "coverage_ceiling_pass": bool(coverage["ceiling_pass"]),
        "target_ceiling_pass": bool(coverage["target_pass"]),
        "gates": strict["legacy_gate"],
        "fixed_ct_denominator": 1228,
        "calibration_thresholds": {m: cal["candidates"][m]["thresholds"] for m in ("cls", "roi")},
        "diagnostic": {
            "public_roi_r1": 0.8218724109362054,
            "public_roi_roc_auc": 0.7558509467208758,
            "public_roi_pr_auc": 0.7280028661204141,
            "public_known_calibration_known": cal["candidates"]["roi"]["grid"]["best"]["metrics"]["known"],
            "devplus_roi_known": strict["candidates"]["roi"]["known_occurrence_acc"],
            "devplus_roi_fixed_ct": strict["candidates"]["roi"]["fixed_ct"]["recall"],
            "public_known_alignment_iou": shift["public_known_bank_role"]["alignment_iou"],
            "devplus_alignment_iou": shift["devplus_supported_known"]["alignment_iou"],
            "public_audit_known_rows": 0,
        },
        "foundation_audit_opened": False,
        "foundation_download_authorized": False,
        "phase16_training_authorized": False,
        "q1_opened": False,
        "reason": "Coverage is sufficient, public matched ROI correspondence and public calibration Known are strong, but frozen DEV+ online Known and fixed-denominator CT fail. The preregistered bounded proposal/input diagnostic supports S-D; it does not authorize a backbone swap, memory redesign, training, or Q1.",
    }
    atomic_json(OUT / "eval/phase15s_decision.json", decision)

    hashes["eval/resource_summary.json"] = {"sha256": sha256(OUT / "eval/resource_summary.json"), "bytes": (OUT / "eval/resource_summary.json").stat().st_size}
    hashes["eval/phase15s_decision.json"] = {"sha256": sha256(OUT / "eval/phase15s_decision.json"), "bytes": (OUT / "eval/phase15s_decision.json").stat().st_size}
    atomic_json(OUT / "manifests/artifact_hashes.json", {"protocol": "trackocd_iclr27_phase15s16", "generated_at": now, "required": hashes, "missing": missing, "symlinked_inputs": symlinks})
    print(json.dumps({"missing": missing, "required_count": len(hashes), "decision": decision["selected_branch"]}, indent=2))


if __name__ == "__main__":
    main()
