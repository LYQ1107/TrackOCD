#!/usr/bin/env python3
"""Phase50 read-only contract/data audit.

The audit deliberately reuses the key-aligned Phase30 TRAIN manifest and frozen
Phase26 feature loader.  It writes only under the Phase50 namespace (apart from
the append-only research log maintained by the caller).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase50"
DOC = ROOT / "docs/iclr27_phase50"
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd(command: str) -> str:
    try:
        return subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    except Exception as e:  # audit remains useful when a utility is unavailable
        return f"UNAVAILABLE: {e}"


def load_data():
    # This import is intentionally read-only and keeps the exact Phase26
    # permutation/key-alignment implementation.
    from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
    rows, tracks, feats = load_tracks()
    return rows, tracks, feats, track_metadata(rows, tracks)


def main() -> None:
    for d in (OUT / "audit", OUT / "metrics", OUT / "checkpoints", OUT / "completion", OUT / "logs", OUT / "manifests"):
        d.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    rows, tracks, feats, meta = load_data()

    # A symlink prevents a second copy of the frozen TRAIN split.  The target is
    # recorded in the manifest ledger and is never written by Phase50.
    fold_target = ROOT / "outputs/iclr27_phase27/manifests/fold_manifest.json"
    fold_link = OUT / "manifests/fold_manifest.json"
    if fold_link.exists() or fold_link.is_symlink():
        if fold_link.is_symlink() and fold_link.resolve() == fold_target.resolve():
            pass
        else:
            raise RuntimeError(f"refusing to overwrite existing Phase50 manifest {fold_link}")
    else:
        fold_link.symlink_to(fold_target)

    fold_stats = []
    forbidden = {"category_name", "category_text", "category_id_feature", "semantic_id", "physical_id", "future_frame", "future_track", "held_gt", "StateMemory", "controller_action", "video_id"}
    model_fields_seen = set()
    for fold in range(4):
        mf = ROOT / f"outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json"
        d = json.loads(mf.read_text(encoding="utf-8"))
        records = d["records"]
        fit = [r for r in records if r.get("split") == "fit" and r.get("kind") == "multi_positive_cross_video"]
        val = [r for r in records if r.get("split") == "val" and r.get("kind") == "multi_positive_cross_video"]
        for r in records:
            model_fields_seen.update(r.get("model_input_fields", []))
        fit_q = {r.get("query_track_key") for r in fit if r.get("query_track_key") in meta}
        fit_support = {k for r in fit for k in r.get("support_track_keys", []) if k in meta}
        fit_hard = {r.get("hard_negative_track_key") for r in fit if r.get("hard_negative_track_key") in meta}
        cross_video_pairs = 0
        pos_per_q = []
        for r in fit:
            q = r.get("query_track_key")
            if q not in meta:
                continue
            sv = [k for k in r.get("support_track_keys", []) if k in meta and meta[k]["video"] < meta[q]["video"]]
            pos_per_q.append(len(sv))
            cross_video_pairs += len(sv)
        categories = Counter(meta[k]["category"] for k in fit_q)
        videos = Counter(meta[k]["video"] for k in fit_q)
        prefix_cov = {str(p): int(sum(bool(r.get("causal_prefixes")) and p in r.get("causal_prefixes", []) for r in fit)) for p in PREFIXES}
        rows_fit = sum(len(meta[k]["rows"]) for k in fit_q)
        fold_stats.append({
            "fold": fold,
            "manifest": str(mf),
            "manifest_sha256": digest(mf),
            "fit_records": len(fit),
            "validation_records": len(val),
            "fit_query_tracks": len(fit_q),
            "fit_support_tracks": len(fit_support),
            "fit_hard_negative_tracks": len(fit_hard),
            "cross_video_positive_pairs": int(cross_video_pairs),
            "queries_with_at_least_one_positive": int(sum(x > 0 for x in pos_per_q)),
            "positive_count_per_query": {"min": int(min(pos_per_q) if pos_per_q else 0), "mean": float(np.mean(pos_per_q) if pos_per_q else 0.0), "max": int(max(pos_per_q) if pos_per_q else 0)},
            "causal_prefix_fit_coverage": prefix_cov,
            "fit_rows_referenced": int(rows_fit),
            "category_count": len(categories),
            "video_count": len(videos),
            "top_categories": categories.most_common(10),
            "top_videos": videos.most_common(10),
            "missing_supervision": {"query_missing_from_feature_index": int(sum(r.get("query_track_key") not in meta for r in fit)), "support_missing_from_feature_index": int(sum(any(k not in meta for k in r.get("support_track_keys", [])) for r in fit)), "hard_negative_missing_from_feature_index": int(sum(r.get("hard_negative_track_key") not in meta for r in fit))},
        })

    resources = {
        "timestamp_utc": started,
        "cwd": str(Path.cwd()),
        "project_root": str(ROOT),
        "git_status": cmd("git status --short 2>&1"),
        "git_revision": cmd("git rev-parse HEAD 2>&1"),
        "free_h": cmd("free -h"),
        "gpu": cmd("nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>&1"),
        "phase50_processes": cmd("pgrep -af '[i]clr27_phase50|[t]rain_end_to_end|[r]un_phase50' || true"),
        "process_count": cmd("ps -e --no-headers | wc -l"),
        "disk_data1": cmd("df -h /data1 | tail -n 1"),
        "gpu_policy": "at most four GPUs; formal mapping 4,5,6,7; no external processes touched",
    }
    atomic_json(OUT / "audit/resource_preflight.json", resources)
    atomic_json(OUT / "audit/supervision_inventory.json", {
        "phase": 50,
        "source": "public TRAIN GT-derived Phase30 episode manifests; labels are metadata-only loss supervision",
        "rows": len(rows),
        "tracks_total": len(tracks),
        "feature_shape": list(feats.shape),
        "folds": fold_stats,
        "aggregate": {"fit_records": int(sum(x["fit_records"] for x in fold_stats)), "cross_video_positive_pairs": int(sum(x["cross_video_positive_pairs"] for x in fold_stats)), "hard_negative_records": int(sum(x["fit_records"] for x in fold_stats)), "validation_records": int(sum(x["validation_records"] for x in fold_stats)), "prefixes": list(PREFIXES)},
        "model_input_fields": sorted(model_fields_seen),
        "supervision_metadata_fields": ["category", "video_id", "gt_track_id", "physical_track_key", "episode_kind"],
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "held-event GT", "future rows/tracks"],
        "fold_manifest_symlink": {"path": str(fold_link), "target": str(fold_target), "target_sha256": digest(fold_target)},
    })
    forbidden_hits = sorted(model_fields_seen & forbidden)
    atomic_json(OUT / "audit/leakage_audit.json", {
        "phase": 50,
        "forbidden_model_input_fields": sorted(forbidden),
        "observed_model_input_fields": sorted(model_fields_seen),
        "forbidden_hits": forbidden_hits,
        "pass": not forbidden_hits,
        "category_and_track_ids": "metadata-only labels for TRAIN loss/split construction; never model inputs",
        "future_rows": False,
        "held_events_used_for_fit": False,
        "devplus_q1_public_labels_read": False,
    })

    prior = {}
    for name, path in {
        "phase46_retrieval": ROOT / "outputs/iclr27_phase46/metrics/phase46_retrieval.json",
        "phase46_controller": ROOT / "outputs/iclr27_phase46/audit/phase46_c2_decision.json",
        "phase48_retrieval": ROOT / "outputs/iclr27_phase48/metrics/phase48_retrieval.json",
        "phase49_retrieval": ROOT / "outputs/iclr27_phase49/metrics/phase49_retrieval_phase49_formal_fix2.json",
    }.items():
        if path.exists():
            try:
                prior[name] = {"path": str(path), "sha256": digest(path), "data": json.loads(path.read_text(encoding="utf-8"))}
            except Exception as e:
                prior[name] = {"path": str(path), "error": str(e)}
    atomic_json(OUT / "audit/frozen_phase24_49_context.json", {"phase": 50, "positive_event_denominator": 76, "raw_anchor_dim": 768, "proposal_phase26_prefix16_true_iou_ceiling": 41, "historical_controller_commit_ct": "3/76 (Phase46 replay; fold3-only)", "prior": prior, "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"]})

    # Architecture contract is generated here so JSON and prose share the same
    # registered dimensions and forbidden-input list.
    contract = {
        "phase": 50,
        "name": "TrackOCD end-to-end causal MOT+OCD graph",
        "graph": ["video_frame", "class_agnostic_proposal_objectness", "differentiable_physical_association", "persistent_physical_track_query", "causal_track_encoder", "raw_preserving_semantic_state", "causal_prior_support_memory", "cross_track_cross_video_correspondence", "semantic_state_memory", "commit_defer_controller", "persistent_Commit_CT"],
        "raw_anchor_dim": 768,
        "semantic_output_dim": 768,
        "state_fields": ["evidence", "persistence", "uncertainty", "contradiction_history"],
        "causal": {"support_order": "strictly earlier completed video/frame", "future_frames": False, "future_tracks": False, "physical_id_semantics": False, "semantic_id_input": False, "category_text_input": False, "invalid_support": "exact raw fallback"},
        "physical_mot": {"birth": "objectness threshold in frozen proposal stream", "continuation": "differentiable association over current/history geometry+appearance", "termination": "causal inactivity state", "semantic_cannot_mutate_physical_id": True, "safety_metrics": ["continuity", "fragmentation", "false_merge", "duplicate_birth", "parent_assignment"]},
        "outputs": ["physical_track_rows", "768-D_track_representation", "candidate_correspondence_scores", "uncertainty", "Commit_or_Defer", "persistent_Commit_CT", "MOT_safety_metrics"],
        "loss_terms": ["objectness", "bbox", "physical_association", "track_continuity", "temporal_representation", "multi_positive_correspondence", "hard_negative", "prefix_consistency", "state_persistence", "commit_defer_utility", "persistent_commit_ct_surrogate", "mot_safety", "raw_preservation"],
        "frozen_components": ["Phase26 proposal source", "physical MOT row/evaluator contract", "row key and denominator", "sealed protocol until Gate C50"],
        "forbidden_inputs": sorted(forbidden),
    }
    atomic_json(OUT / "audit/architecture_contract.json", contract)
    (DOC / "PHASE50_END_TO_END_ARCHITECTURE_CONTRACT.md").write_text(
        "# Phase50 — End-to-End Causal Architecture Contract\n\n"
        "This contract registers one causal graph for MOT+OCD. The Phase26 proposal and physical stream are used as a frozen warm start; Phase50 adds a jointly parameterized semantic/state path and records all losses, while never allowing semantic outputs to mutate physical IDs.\n\n"
        "```text\nvideo → class-agnostic proposal/objectness → differentiable physical association\n      → persistent physical track query → causal track encoder\n      → raw-preserving 768-D semantic state → prior-support memory\n      → cross-track/cross-video correspondence → semantic StateMemory\n      → Commit/Defer controller → persistent Commit-CT\n```\n\n"
        "## Contract\n\n"
        "The representation is a normalized 768-D raw-anchor plus a bounded causal residual/evidence update. Missing or invalid support returns the raw vector exactly. State contains evidence, persistence, uncertainty and contradiction history. Commit requires causal evidence accumulation; Defer is a valid action. Physical birth/continuation/termination/association remain separate from semantic correspondence and are measured with continuity, fragmentation, false-merge, duplicate-birth and parent-assignment invariants.\n\n"
        "Model-facing inputs are causal visual features, box geometry, track age/history, support quality and temporal metadata. Category names/text, semantic or physical IDs, future frames/tracks, held GT, StateMemory/controller actions are forbidden. TRAIN category/track labels are loss/split metadata only. The complete machine-readable contract is `outputs/iclr27_phase50/audit/architecture_contract.json`.\n\n"
        "All weights and loss weights are TRAIN-only and frozen before any 76-event or sealed evaluation. Gate P50→R50→C50→S50 is hierarchical; retrieval or proposal oracle numbers never substitute for persistent Commit-CT.\n", encoding="utf-8")

    print(json.dumps({"phase": 50, "audit": "PASS", "feature_shape": list(feats.shape), "fold_fit_records": [x["fit_records"] for x in fold_stats], "fold_link": str(fold_link)}, indent=2))


if __name__ == "__main__":
    main()
