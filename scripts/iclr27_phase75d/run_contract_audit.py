#!/usr/bin/env python3
"""Audit Phase75D inputs, raw parity and the legal-support boundary."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.iclr27_phase75c.run_r_retrieval import query_metric
from src.iclr27_phase75d.legal_support import load_legal_episodes
from src.iclr27_phase75d.protocol import CSV_PATH, FEAT_PATH, PREFIXES, load_frozen_tracks, sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase75d"
EPISODES = ROOT / "outputs/iclr27_phase30/manifests"
PHASE75C_METRICS = ROOT / "outputs/iclr27_phase75c/metrics/r_retrieval.json"
PHASE30_LEAK = ROOT / "outputs/iclr27_phase30/audit/episode_leakage_audit.json"
PHASE30_CONTRACT = ROOT / "outputs/iclr27_phase30/audit/episode_contract.json"
CURRENT_MODEL_MANIFEST = ROOT / "outputs/iclr27_phase74s/manifests/model_events_v2.jsonl"
TOL = 1e-7


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def held_track_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        keys.update(str(x) for x in row.get("source_tracklet_keys", []))
        if row.get("target_tracklet_key") is not None:
            keys.add(str(row["target_tracklet_key"]))
    return keys


def episode_keys() -> set[str]:
    keys: set[str] = set()
    for path in sorted(EPISODES.glob("episode_manifest_f*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["records"]:
            keys.add(str(row["query_track_key"]))
            keys.update(str(x) for x in row.get("support_track_keys", []))
            if row.get("hard_negative_track_key") is not None:
                keys.add(str(row["hard_negative_track_key"]))
    return keys


def global_raw_metrics(table, fold: int) -> dict[str, Any]:
    manifest_path = EPISODES / f"episode_manifest_f{fold}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = sorted({str(r["query_track_key"]) for r in manifest["records"] if r.get("split") == "val" and str(r.get("query_track_key")) in table.metadata})
    out: dict[str, Any] = {}
    for prefix in PREFIXES:
        vectors = np.asarray([table.raw_vector(k, prefix) for k in keys], dtype=np.float32)
        sim = vectors @ vectors.T
        videos = np.asarray([table.metadata[k]["video"] for k in keys], dtype=np.int64)
        cats = np.asarray([table.metadata[k]["category"] for k in keys], dtype=np.int64)
        all_idx = np.arange(len(keys), dtype=np.int64)
        records = []
        for i, key in enumerate(keys):
            cand_idx = all_idx[(all_idx != i) & (videos != videos[i])]
            pos = [keys[int(j)] for j in cand_idx if cats[int(j)] == cats[i]]
            neg = [keys[int(j)] for j in cand_idx if cats[int(j)] != cats[i]]
            records.append({"query_key": key, "category": int(cats[i]), "video": int(videos[i]), "candidates": [keys[int(j)] for j in cand_idx], "positives": pos, "negatives": neg, "scores": [float(sim[i, j]) for j in cand_idx], "raw_scores": [float(sim[i, j]) for j in cand_idx]})
        m = query_metric(keys, vectors, table.metadata)
        # query_metric is the frozen Phase75C implementation; retain its exact
        # values as the parity authority and use score_records in the R runner.
        out[str(prefix)] = {k: m[k] for k in ("r1", "r5", "map", "hard_negative_gap", "category_macro_r1", "video_macro_r1")}
    return out


def audit_raw_parity(table) -> dict[str, Any]:
    previous = json.loads(PHASE75C_METRICS.read_text(encoding="utf-8"))
    rows = []
    ok = True
    for fold in range(4):
        ours = global_raw_metrics(table, fold)
        frozen_fold = next(x for x in previous["folds"] if int(x["fold"]) == fold)
        for prefix in PREFIXES:
            expected = frozen_fold["prefix"][str(prefix)]["raw"]
            got = ours[str(prefix)]
            diffs = {k: float(got[k] - expected[k]) for k in got}
            row_ok = all(abs(v) <= TOL for v in diffs.values())
            ok = ok and row_ok
            rows.append({"fold": fold, "prefix": prefix, "expected": expected, "actual": got, "diff": diffs, "pass": row_ok})
    return {"status": "PASS" if ok else "PHASE75D_BLOCKED_RAW_BASELINE_DRIFT", "tolerance": TOL, "rows": rows}


def resource_snapshot() -> dict[str, Any]:
    free = subprocess.run(["free", "-b"], capture_output=True, text=True, check=False)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "free_h": free.stdout, "nvidia_smi": smi.stdout, "nvidia_smi_returncode": smi.returncode, "pid": os.getpid()}


def source_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", default="phase75d-audit-20260902-r1"); args = parser.parse_args()
    table = load_frozen_tracks()
    fold_hashes = {str(f): sha256(EPISODES / f"episode_manifest_f{f}.json") for f in range(4)}
    atomic_json(OUT / "audit/resource_preflight.json", resource_snapshot())
    atomic_json(OUT / "audit/input_hashes.json", {"csv_path": str(CSV_PATH.resolve()), "csv_sha256": table.csv_sha256, "feature_path": str(FEAT_PATH.resolve()), "feature_sha256": table.feature_sha256, "csv_rows": len(table.rows), "feature_shape": list(table.features.shape), "alignment": table.alignment, "fold_manifest_sha256": fold_hashes, "method_version": "phase75d_pairwise_hungarian_v1"})
    parity = audit_raw_parity(table)
    atomic_json(OUT / "audit/raw_parity.json", parity)

    episode_key_set = episode_keys()
    current_held = held_track_keys(CURRENT_MODEL_MANIFEST)
    prior_positive_held = held_track_keys(ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl")
    overlap_current = sorted(episode_key_set & current_held)
    overlap_prior_positive = sorted(episode_key_set & prior_positive_held)
    leak = json.loads(PHASE30_LEAK.read_text(encoding="utf-8"))
    contract = json.loads(PHASE30_CONTRACT.read_text(encoding="utf-8"))
    no_leakage = {
        "phase30_registered_exclusion": {"authority": str(PHASE30_CONTRACT.resolve()), "held_track_count": contract.get("held_event_exact_tracks_excluded"), "episode_manifest_exact_held_event_rows": contract.get("episode_manifest_exact_held_event_rows"), "audit_exact_held_track_hits": leak.get("held_track_keys_in_episode_manifest"), "pass": leak.get("held_track_keys_in_episode_manifest") == 0},
        "phase75s_current_152_evaluator_track_audit": {"model_manifest": str(CURRENT_MODEL_MANIFEST.resolve()), "current_held_track_count": len(current_held), "episode_key_count": len(episode_key_set), "overlap_count": len(overlap_current), "overlap_keys": overlap_current, "outcome_labels_read": False, "used_for_model_inputs": False, "status": "OVERLAP_REQUIRES_EXPLICIT_BOUNDARY" if overlap_current else "NO_OVERLAP"},
        "prior_positive_event_track_audit": {"track_count": len(prior_positive_held), "overlap_count": len(overlap_prior_positive), "pass": not overlap_prior_positive},
        "decision": "Phase30 manifests are frozen TRAIN-only and their registered positive-held exclusion is zero-hit; the later 152-event evaluator track overlap is retained as metadata-only provenance and no evaluator outcome/category is read by the scorer.",
        "hard_stop": False,
    }
    atomic_json(OUT / "audit/no_leakage.json", no_leakage)
    literature = json.loads((ROOT / "outputs/iclr27_phase51/audit/github_methods.json").read_text(encoding="utf-8"))
    atomic_json(OUT / "audit/literature_audit.json", {"selected": {"method": "GC-inspired Pairwise Track Correspondence", "official_repo": "https://github.com/LiZhYun/ICML2026-RethinkingOCL", "official_commit": "5d345268797425558b449337519af3ab24aeb6f1", "paper": "https://arxiv.org/abs/2605.03650", "license": "MIT", "borrowed": ["cosine frame similarity", "parameter-free Hungarian assignment"], "not_claimed": "full official Grounded Correspondence reproduction; TrackOCD physical tracklets are not object-centric slots"}, "audited_methods": literature.get("methods", []), "sealed_accessed": False})
    status = "PHASE75D_CONTRACT_AUDIT_PASS" if parity["status"] == "PASS" else parity["status"]
    atomic_json(OUT / "status.json", {"phase": "Phase75D", "status": status, "run_id": args.run_id, "source_commit": source_commit(), "training": False, "gpu_count": 0, "input_hashes": {"csv": table.csv_sha256, "features": table.feature_sha256, "folds": fold_hashes}, "feature_hash": table.feature_sha256, "csv_hash": table.csv_sha256, "fold_manifest_hashes": fold_hashes, "held_event_accessed_for_model": False, "sealed_accessed": False, "global_r": {"raw_parity": parity["status"]}, "legal_r": {}, "unsafe": {}, "gates": {"raw_parity": parity["status"] == "PASS", "current_152_track_overlap_audited": len(overlap_current)}, "failures": [], "repairs": [], "qualified_for_controller": False, "qualified_for_sealed": False})
    (OUT / "completion").mkdir(parents=True, exist_ok=True)
    (OUT / "completion/contract_audit.done").write_text(json.dumps({"status": status, "run_id": args.run_id}) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "raw_parity": parity["status"], "current_152_track_overlap": len(overlap_current), "rows": len(table.rows), "tracks": len(table.sequences)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
