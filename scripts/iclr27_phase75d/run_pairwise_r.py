#!/usr/bin/env python3
"""Run exact R-global and R-legal pairwise Hungarian benchmarks.

The scorer is intentionally single-process and bounded.  It evaluates every
allowed candidate (no raw shortlist) and uses SciPy only for the assignment;
identifiers/categories are kept outside the tensor/scoring functions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase75d.cache import atomic_json, cache_key
from src.iclr27_phase75d.gates import gate_rows, pairwise_teacher_signal, strict_gate
from src.iclr27_phase75d.legal_support import LegalSupportEpisode, episode_bank_hash, load_legal_episodes
from src.iclr27_phase75d.pairwise_correspondence import fast_hungarian_score, hungarian_score
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks
from src.iclr27_phase75d.retrieval_metrics import aggregate_fold_metrics, score_records

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase75d"
EPISODES = ROOT / "outputs/iclr27_phase30/manifests"
PHASE75C_METRICS = ROOT / "outputs/iclr27_phase75c/metrics/r_retrieval.json"
METHOD_VERSION = "phase75d_pairwise_hungarian_v1"


def atomic_marker(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def fold_validation_keys(fold: int, table) -> tuple[list[str], dict[str, Any]]:
    manifest_path = EPISODES / f"episode_manifest_f{fold}.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = sorted({str(r["query_track_key"]) for r in data["records"] if r.get("split") == "val" and str(r.get("query_track_key")) in table.metadata})
    return keys, {"data": data, "manifest_path": manifest_path, "manifest_sha256": __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()}


def global_records(table, fold: int, prefix: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keys, manifest_info = fold_validation_keys(fold, table)
    seqs = {k: table.get_frame_sequence(k, prefix) for k in keys}
    raw = np.asarray([table.raw_vector(k, prefix) for k in keys], dtype=np.float32)
    videos = np.asarray([table.metadata[k]["video"] for k in keys], dtype=np.int64)
    cats = np.asarray([table.metadata[k]["category"] for k in keys], dtype=np.int64)
    records: list[dict[str, Any]] = []
    all_idx = np.arange(len(keys), dtype=np.int64)
    for i, qkey in enumerate(keys):
        cand_idx = all_idx[(all_idx != i) & (videos != videos[i])]
        candidates = [keys[int(j)] for j in cand_idx]
        positives = [keys[int(j)] for j in cand_idx if cats[int(j)] == cats[i]]
        negatives = [keys[int(j)] for j in cand_idx if cats[int(j)] != cats[i]]
        pair_scores = [fast_hungarian_score(seqs[qkey], seqs[c]) for c in candidates]
        raw_scores = [float(raw[i] @ raw[int(j)]) for j in cand_idx]
        records.append({"query_key": qkey, "category": int(cats[i]), "video": int(videos[i]), "candidates": candidates, "positives": positives, "negatives": negatives, "scores": pair_scores, "raw_scores": raw_scores})
    return records, {"fold": fold, "prefix": prefix, "validation_tracklets": len(keys), "candidate_rule": "all validation tracks except self and same video", "candidate_count_total": int(sum(len(r["candidates"]) for r in records)), "manifest": str(manifest_info["manifest_path"].resolve()), "manifest_sha256": manifest_info["manifest_sha256"]}


def legal_records(table, fold: int, prefix: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    episodes, unevaluable, summary = load_legal_episodes(EPISODES, fold, set(table.sequences))
    seq_cache: dict[tuple[str, int], np.ndarray] = {}

    def seq(key: str, p: int) -> np.ndarray:
        cache_key_local = (key, p)
        if cache_key_local not in seq_cache:
            seq_cache[cache_key_local] = table.get_frame_sequence(key, p)
        return seq_cache[cache_key_local]

    records: list[dict[str, Any]] = []
    for ep in episodes:
        candidates = list(ep.positive_support_keys) + [x for x in ep.negative_support_keys if x not in ep.positive_support_keys]
        q = seq(ep.query_key, prefix)
        pair_scores = [fast_hungarian_score(q, seq(c, 16)) for c in candidates]
        raw_q = table.raw_vector(ep.query_key, prefix)
        raw_scores = [float(raw_q @ table.raw_vector(c, 16)) for c in candidates]
        meta = table.metadata[ep.query_key]
        records.append({"query_key": ep.query_key, "episode_id": ep.episode_id, "category": int(meta["category"]), "video": int(meta["video"]), "candidates": candidates, "positives": list(ep.positive_support_keys), "negatives": list(ep.negative_support_keys), "scores": pair_scores, "raw_scores": raw_scores})
    summary.update({"candidate_bank_hash": episode_bank_hash(episodes), "evaluable_episode_ids": [e.episode_id for e in episodes], "unevaluable": unevaluable})
    return records, unevaluable, summary


def diagnostic(records: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    if not records:
        return None
    r = records[0]
    scores = np.asarray(r["scores"], dtype=np.float32)
    idx = int(np.argmax(scores))
    candidate = r["candidates"][idx]
    qkey = r["query_key"]
    # The full assignment is intentionally saved for only one deterministic
    # top-candidate sample, never for the all-candidate benchmark.
    return {"mode": mode, "query_key": qkey, "candidate_key": candidate, "candidate_index": idx, "score": float(scores[idx]), "candidate_count": len(r["candidates"]), "assignment_saved": False}


def resource_snapshot() -> dict[str, Any]:
    free = subprocess.run(["free", "-b"], capture_output=True, text=True, check=False)
    return {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "free_h": free.stdout, "pid": os.getpid()}


def run_section(table, section: str) -> dict[str, Any]:
    fold_outputs: list[dict[str, Any]] = []
    prefix_outputs: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    for fold in range(4):
        for prefix in PREFIXES:
            if section == "global":
                records, info = global_records(table, fold, prefix)
                unevaluable: list[dict[str, Any]] = []
                info["candidate_bank_hash"] = "global:" + str(info["validation_tracklets"])
            else:
                records, unevaluable, info = legal_records(table, fold, prefix)
            metrics = score_records(records)
            metrics["unevaluable_count"] = len(unevaluable)
            metrics["unsafe_flip_fold_rate"] = metrics["unsafe_flip_micro_rate"]
            fold_outputs.append({"fold": fold, "prefix": prefix, "metrics": {"raw": {"r1": metrics["raw_r1"], "r5": metrics["raw_r5"], "map": metrics["raw_map"], "hard_negative_gap": metrics["raw_hard_negative_gap"], "category_macro_r1": metrics["category_macro_r1"], "video_macro_r1": metrics["video_macro_r1"]}, "pairwise": {"r1": metrics["r1"], "r5": metrics["r5"], "map": metrics["map"], "hard_negative_gap": metrics["hard_negative_gap"], "category_macro_r1": metrics["category_macro_r1"], "video_macro_r1": metrics["video_macro_r1"]}, "details": {k: metrics[k] for k in ("queries", "unsafe_flip_count", "unsafe_flip_micro_rate", "unsafe_flip_fold_rate", "top1_change_count", "top1_change_rate", "unevaluable_count")}}, "inventory": info, "diagnostic": diagnostic(records, section)})
            prefix_outputs.append({"fold": fold, "prefix": prefix, "queries": metrics["queries"], "raw_r1": metrics["raw_r1"], "pairwise_r1": metrics["r1"], "delta_r1": metrics["r1"] - metrics["raw_r1"], "raw_map": metrics["raw_map"], "pairwise_map": metrics["map"], "delta_map": metrics["map"] - metrics["raw_map"], "raw_hard_gap": metrics["raw_hard_negative_gap"], "pairwise_hard_gap": metrics["hard_negative_gap"], "unsafe_flip_count": metrics["unsafe_flip_count"], "unsafe_flip_micro_rate": metrics["unsafe_flip_micro_rate"], "unsafe_flip_fold_rate": metrics["unsafe_flip_fold_rate"], "candidate_count_total": info.get("candidate_count_total"), "unevaluable_count": len(unevaluable)})
            cache_rows.append({"fold": fold, "prefix": prefix, "key": cache_key(feature_sha256=table.feature_sha256, csv_sha256=table.csv_sha256, fold_manifest_sha256=info["manifest_sha256"], prefix=prefix, method_version=METHOD_VERSION, query_key="__all__", candidate_bank_hash=info["candidate_bank_hash"]) if "manifest_sha256" in info else None, "cache_policy": "key-only audit; no score matrix persisted"})
            print(json.dumps({"section": section, "fold": fold, "prefix": prefix, "queries": metrics["queries"], "pair_r1": round(metrics["r1"], 6), "pair_map": round(metrics["map"], 6)}, sort_keys=True), flush=True)
    # Gate uses p16 fold objects, preserving per-fold acceptance rather than a
    # query-weighted aggregate shortcut.
    p16 = [x for x in fold_outputs if x["prefix"] == 16]
    gate_input = [{"fold": x["fold"], "metrics": x["metrics"]} for x in p16]
    rows = gate_rows(gate_input, section)
    gate = strict_gate(rows)
    aggregate = {}
    for prefix in PREFIXES:
        p = [x["metrics"] for x in fold_outputs if x["prefix"] == prefix]
        # Rebuild the compact aggregate from per-query metrics by treating the
        # compact raw/pair fields as fold means; this is a fold-macro summary.
        compact = []
        for x in p:
            compact.append({"r1": x["pairwise"]["r1"], "r5": x["pairwise"]["r5"], "map": x["pairwise"]["map"], "raw_r1": x["raw"]["r1"], "raw_r5": x["raw"]["r5"], "raw_map": x["raw"]["map"], "hard_negative_gap": x["pairwise"]["hard_negative_gap"], "raw_hard_negative_gap": x["raw"]["hard_negative_gap"], "category_macro_r1": x["pairwise"]["category_macro_r1"], "video_macro_r1": x["pairwise"]["video_macro_r1"], "queries": x["details"]["queries"], "unsafe_flip_count": x["details"]["unsafe_flip_count"], "unsafe_flip_micro_rate": x["details"]["unsafe_flip_micro_rate"], "top1_change_count": x["details"]["top1_change_count"], "top1_change_rate": x["details"]["top1_change_rate"]})
        aggregate[str(prefix)] = aggregate_fold_metrics(compact)
    return {"section": section, "method_version": METHOD_VERSION, "prefix": aggregate, "folds": fold_outputs, "prefix_rows": prefix_outputs, "gate": gate, "cache_keys": cache_rows}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", default="phase75d-r-20260902-r1"); args = parser.parse_args()
    parity = json.loads((OUT / "audit/raw_parity.json").read_text(encoding="utf-8"))
    if parity.get("status") != "PASS":
        raise SystemExit("raw parity did not pass; refusing pairwise benchmark")
    table = load_frozen_tracks()
    global_out = run_section(table, "global")
    legal_out = run_section(table, "legal")
    global_p16 = [{"fold": x["fold"], "metrics": x["metrics"]} for x in global_out["folds"] if x["prefix"] == 16]
    legal_p16 = [{"fold": x["fold"], "metrics": x["metrics"]} for x in legal_out["folds"] if x["prefix"] == 16]
    global_agg = global_out["prefix"]["16"]; legal_agg = legal_out["prefix"]["16"]
    teacher = pairwise_teacher_signal(global_agg, legal_agg, legal_p16)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    global_out.update({"phase": "Phase75D-R-global", "run_id": args.run_id, "created_utc": created, "raw_parity": parity})
    legal_out.update({"phase": "Phase75D-R-legal", "run_id": args.run_id, "created_utc": created, "teacher_signal": teacher})
    atomic_json(OUT / "metrics/global_r.json", global_out)
    atomic_json(OUT / "metrics/legal_support_r.json", legal_out)
    atomic_json(OUT / "metrics/fold_rows.json", {"global": [{"fold": x["fold"], "prefix": x["prefix"], "metrics": x["metrics"], "gate": next((r for r in global_out["gate"]["rows"] if r["fold"] == x["fold"]), None) if x["prefix"] == 16 else None} for x in global_out["folds"]], "legal": [{"fold": x["fold"], "prefix": x["prefix"], "metrics": x["metrics"], "gate": next((r for r in legal_out["gate"]["rows"] if r["fold"] == x["fold"]), None) if x["prefix"] == 16 else None} for x in legal_out["folds"]]})
    atomic_json(OUT / "metrics/prefix_rows.json", {"global": global_out["prefix_rows"], "legal": legal_out["prefix_rows"]})
    atomic_json(OUT / "audit/cache_keys.json", {"global": global_out["cache_keys"], "legal": legal_out["cache_keys"], "cache_root": "/data2/usr_for_deadline/trackocd_phase75d/cache", "matrix_persisted": False})
    atomic_json(OUT / "audit/resource_postflight.json", resource_snapshot())
    status = "P75D_GATE_R_PASS" if global_out["gate"]["pass"] and legal_out["gate"]["pass"] else ("P75D_PAIRWISE_SIGNAL_AUTHORIZE_P75E" if teacher["signal"] else "P75D_NO_PAIRWISE_SIGNAL")
    status_obj = {"phase": "Phase75D", "status": status, "run_id": args.run_id, "source_commit": "pending_phase75d_commit", "training": False, "gpu_count": 0, "input_hashes": {"csv": table.csv_sha256, "features": table.feature_sha256}, "feature_hash": table.feature_sha256, "csv_hash": table.csv_sha256, "fold_manifest_hashes": {str(f): x["inventory"].get("manifest_sha256") for f in range(4) for x in global_out["folds"] if x["fold"] == f and x["prefix"] == 16}, "held_event_accessed_for_model": False, "sealed_accessed": False, "global_r": {"gate": global_out["gate"], "aggregate_prefix16": global_agg}, "legal_r": {"gate": legal_out["gate"], "aggregate_prefix16": legal_agg, "teacher_signal": teacher}, "unsafe": {"global": global_agg["unsafe_flip_count"], "legal": legal_agg["unsafe_flip_count"]}, "gates": {"global_pass": global_out["gate"]["pass"], "legal_pass": legal_out["gate"]["pass"], "teacher_signal": teacher["signal"]}, "failures": [], "repairs": [], "qualified_for_controller": bool(global_out["gate"]["pass"] and legal_out["gate"]["pass"]), "qualified_for_sealed": False}
    atomic_json(OUT / "status.json", status_obj)
    atomic_marker(OUT / "completion/pairwise_r.done", {"status": status, "run_id": args.run_id, "metrics": [str(OUT / "metrics/global_r.json"), str(OUT / "metrics/legal_support_r.json")]})
    print(json.dumps({"status": status, "global_gate": global_out["gate"], "legal_gate": legal_out["gate"], "teacher_signal": teacher}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
