#!/usr/bin/env python3
"""Single fixed-observed-step re-evaluation of the Phase86 RF diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87"
EP = ROOT / "outputs" / "iclr27_phase30" / "manifests"
LINEAGE = Path("/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/temporal_mean_full/full_temporal_lineage.jsonl")
UNIONS = Path("/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/temporal_mean_full/union_events.jsonl")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iclr27_phase23.protocol import load_rows
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks
from src.iclr27_phase75d.retrieval_metrics import score_records
from src.iclr27_phase87.causal_root_fragments_fixed import make_builder


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atom(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def chamfer(query: np.ndarray, candidate: np.ndarray) -> float:
    query = query if len(query) else np.zeros((1, 768), np.float32)
    candidate = candidate if len(candidate) else np.zeros((1, 768), np.float32)
    similarity = query @ candidate.T
    return float(0.5 * (np.max(similarity, axis=1).mean() + np.max(similarity, axis=0).mean()))


def fold_keys(fold: int) -> list[str]:
    manifest = json.loads((EP / f"episode_manifest_f{fold}.json").read_text())
    return sorted({str(row["query_track_key"]) for row in manifest["records"] if row.get("split") == "val"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="phase86_rf_causal_correction")
    args = parser.parse_args()
    table = load_frozen_tracks()
    public = load_rows()
    builder, builder_audit = make_builder(table, public, LINEAGE, UNIONS)
    fold_rows = []
    for fold in range(4):
        keys = [key for key in fold_keys(fold) if key in table.metadata]
        videos = np.asarray([table.metadata[key]["video"] for key in keys])
        categories = np.asarray([table.metadata[key]["category"] for key in keys])
        for prefix in PREFIXES:
            cache = {(key, prefix): builder.build(key, prefix) for key in keys}
            cache.update({(key, 16): builder.build(key, 16) for key in keys})
            records = []
            for index, query_key in enumerate(keys):
                candidate_indices = np.where((np.arange(len(keys)) != index) & (videos != videos[index]))[0]
                candidates = [keys[int(j)] for j in candidate_indices]
                positives = [keys[int(j)] for j in candidate_indices if categories[int(j)] == categories[index]]
                negatives = [keys[int(j)] for j in candidate_indices if categories[int(j)] != categories[index]]
                query = cache[(query_key, prefix)]
                scores, raw_scores = [], []
                for candidate_key in candidates:
                    candidate = cache[(candidate_key, 16)]
                    raw = float(query["raw_anchor_vector"] @ candidate["raw_anchor_vector"])
                    raw_scores.append(raw)
                    scores.append(raw + 0.05 * np.tanh(chamfer(query["fragment_vectors"], candidate["fragment_vectors"]) - raw))
                records.append({"query_key": query_key, "category": int(categories[index]), "video": int(videos[index]), "candidates": candidates, "positives": positives, "negatives": negatives, "scores": scores, "raw_scores": raw_scores})
            metrics = score_records(records)
            fold_rows.append({"fold": fold, "prefix": prefix, "metrics": metrics})
    aggregate = []
    for prefix in PREFIXES:
        rows = [row["metrics"] for row in fold_rows if row["prefix"] == prefix]
        aggregate.append({"prefix": prefix, "aggregate": {"r1": float(np.mean([row["r1"] for row in rows])), "raw_r1": float(np.mean([row["raw_r1"] for row in rows])), "map": float(np.mean([row["map"] for row in rows])), "raw_map": float(np.mean([row["raw_map"] for row in rows])), "hard_negative_gap": float(np.mean([row["hard_negative_gap"] for row in rows])), "raw_hard_negative_gap": float(np.mean([row["raw_hard_negative_gap"] for row in rows])), "queries": int(sum(row["queries"] for row in rows)), "unsafe_flip_count": int(sum(row["unsafe_flip_count"] for row in rows))}})
    p16 = [row for row in fold_rows if row["prefix"] == 16]
    p16_aggregate = next(row["aggregate"] for row in aggregate if row["prefix"] == 16)
    non_decreasing = sum(int(row["metrics"]["r1"] >= row["metrics"]["raw_r1"] and row["metrics"]["map"] >= row["metrics"]["raw_map"]) for row in p16)
    decision = "RF_PHASE86_OBSERVED_STEP_CORRECTION_PASS_TRAIN_VALIDATION" if non_decreasing >= 3 and p16_aggregate["unsafe_flip_count"] == 0 else "RF_PHASE86_OBSERVED_STEP_CORRECTION_NEGATIVE"
    result = {"schema_version": "trackocd.phase87.rf_causal_correction.v1", "phase": 87, "route": "PHASE86_RF_OBSERVED_STEP_CORRECTION", "tag": args.tag, "fold_rows": fold_rows, "aggregate": aggregate, "p16_non_decreasing_folds": non_decreasing, "decision": decision, "builder_audit": builder_audit, "inputs": {"lineage": str(LINEAGE), "lineage_sha256": sha(LINEAGE), "union_events": str(UNIONS), "union_events_sha256": sha(UNIONS)}, "config": {"residual_scale": 0.05, "query_prefixes": list(PREFIXES), "candidate_prefix": 16, "step_rule": "sorted(frame_id,image_id) ordinal per video"}, "training": False, "diagnostic_only": True, "controller_run": False, "sealed_run": False, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_or_text_as_model_input": False}
    atom(OUT / "audit" / "phase86_rf_causal_correction.json", result)
    atom(OUT / "metrics" / f"{args.tag}.json", result)
    atom(OUT / "completion" / f"{args.tag}.done", {"status": "DONE", "metrics": str((OUT / "metrics" / f"{args.tag}.json").resolve()), "decision": decision})
    print(json.dumps({"decision": decision, "p16": p16_aggregate, "non_decreasing_folds": non_decreasing}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
