#!/usr/bin/env python3
"""Frozen Phase86 D0--D3 diagnostic OCD replay.

Only D0 (native Phase19R feature stream) and D1 (Phase85 temporal physical
vectors) have a legal 768-D row-vector adapter.  D2/D3 are recorded as
interface-incompatible because their frozen artifacts contain score-level
candidate decisions, not a causal per-row 768-D vector accepted by RC-MS-OCD.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate


OUT = ROOT / "outputs/iclr27_phase86"
PREFIXES = (1, 2, 4, 8, 16)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atom_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


class VectorData(Phase19RData):
    """Replace only the causal visual vector; preserve geometry/quality/state."""

    def __init__(self, fold: int, vectors: np.ndarray, keys: list[str]):
        super().__init__(fold=fold, final=False)
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._key_index = {str(k): i for i, k in enumerate(keys)}
        missing = [k for k in self._key_index if k not in self.track_rows]
        if missing:
            raise RuntimeError(f"vector stream has unknown track keys: {missing[:3]}")

    @staticmethod
    def _prefix_index(position: int) -> int:
        n = int(position) + 1
        if n <= 1: return 0
        if n <= 2: return 1
        if n <= 4: return 2
        if n <= 8: return 3
        return 4

    def prefix(self, track_key: str, position: int | None = None):
        raw, geom, quality, pos = super().prefix(track_key, position)
        j = self._key_index.get(str(track_key))
        if j is None:
            return raw, geom, quality, pos
        v = self._vectors[self._prefix_index(pos), j]
        v = np.asarray(v, dtype=np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-8)
        return v, geom, quality, pos


def replay(label: str, vector_path: Path | None, checkpoint_root: Path) -> dict[str, Any]:
    if vector_path is None:
        keys = None; vectors = None; source_sha = None
    else:
        z = np.load(vector_path, allow_pickle=False)
        keys = [str(x) for x in z["keys"].tolist()]
        vectors = np.asarray(z["vectors"], dtype=np.float32)
        if vectors.shape != (5, len(keys), 768):
            raise RuntimeError(f"bad {label} vector shape {vectors.shape}")
        source_sha = sha(vector_path)
    folds = []
    all_records = []
    for fold in range(4):
        data = Phase19RData(fold=fold, final=False) if vectors is None else VectorData(fold, vectors, keys)
        ck = checkpoint_root / f"fold{fold}_best_internal.pt"
        result = evaluate_candidate("main", data, ck, torch.device("cpu"))
        folds.append({"fold": fold, "checkpoint": str(ck), "checkpoint_sha256": sha(ck),
                      "metrics": result["metrics"], "known_metrics": result["known_metrics"],
                      "events": result["events"]})
        for rec in result["records"]:
            all_records.append({"stream": label, **rec})
    total_pos = sum(int(x["metrics"]["commit_ct"]["correct"]) for x in folds)
    total_elig = sum(int(x["metrics"]["commit_ct"]["eligible"]) for x in folds)
    summary = {
        "stream": label, "adapter": "native Phase19R prefix" if vector_path is None else "Phase85 prefix-vector deterministic mapping",
        "vector_source": None if vector_path is None else str(vector_path.resolve()), "vector_source_sha256": source_sha,
        "folds": folds, "aggregate_commit_ct": {"correct": total_pos, "eligible": total_elig, "recall": total_pos / max(total_elig, 1)},
        "controller_frozen": True, "thresholds_frozen": True, "state_memory_frozen": True,
        "diagnostic_only": True, "diagnostic_label": "DIAGNOSTIC_ONLY_DO_NOT_SELECT",
        "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
    }
    return {"summary": summary, "records": all_records}


def main() -> None:
    ckroot = ROOT / "outputs/iclr27_phase19r/checkpoints"
    q0 = ROOT / "outputs/iclr27_phase85/manifests/physical_r_q0_q0_parity_v5_vectors.npz"
    temporal = ROOT / "outputs/iclr27_phase85/manifests/physical_r_improved_improved_single_anchor_v2_vectors.npz"
    streams = {
        "D0_historical_Q0_RCMSOCD": replay("D0_historical_Q0_RCMSOCD", None, ckroot),
        "D1_temporal_physical_RCMSOCD": replay("D1_temporal_physical_RCMSOCD", temporal, ckroot),
    }
    incompatible = {
        "D2_raw_source_conditioned_support": {
            "status": "DIAGNOSTIC_OCD_STREAM_NOT_INTERFACE_COMPATIBLE",
            "reason": "Phase85 raw source-conditioned artifact is candidate score/reliable-event evidence, not a causal per-row 768-D vector accepted by the frozen RC-MS-OCD input contract",
            "artifact": str((ROOT / "outputs/iclr27_phase85/metrics/support_event_replay.json").resolve()),
            "artifact_sha256": sha(ROOT / "outputs/iclr27_phase85/metrics/support_event_replay.json"),
        },
        "D3_bounded_reranker_only_support": {
            "status": "DIAGNOSTIC_OCD_STREAM_NOT_INTERFACE_COMPATIBLE",
            "reason": "Phase85 bounded reranker artifact exposes score-level selected candidate records and has no controller-compatible causal row-vector exporter; no score rescale or invented adapter is allowed",
            "artifact": str((ROOT / "outputs/iclr27_phase85/metrics/support_event_replay.json").resolve()),
            "artifact_sha256": sha(ROOT / "outputs/iclr27_phase85/metrics/support_event_replay.json"),
        },
    }
    traces = []
    for v in streams.values(): traces.extend(v["records"])
    summary = {
        "schema_version": "trackocd.phase86.diagnostic_ocd.v1", "phase": 86,
        "label": "DIAGNOSTIC_ONLY_DO_NOT_SELECT", "denominator": {"positive_events": 76, "negative_events": 76, "prefixes": list(PREFIXES)},
        "streams": {k: v["summary"] for k, v in streams.items()}, "incompatible_streams": incompatible,
        "controller_manifest": str((OUT / "manifests/frozen_controller_manifest.json").resolve()),
        "metric_definition_source": str((ROOT / "src/iclr27_phase19r/evaluation/internal.py").resolve()),
        "diagnostic_results_not_consumed_by_training": True,
        "public_dev_q1_sealed_accessed": False, "sealed_run": False,
    }
    atom_json(OUT / "diagnostic_ocd/summary.json", summary)
    trace_path = OUT / "diagnostic_ocd/event_traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = trace_path.with_name(trace_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in traces:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())
    tmp.replace(trace_path)
    atom_json(OUT / "completion/diagnostic_ocd.done", {"status": "DONE", "summary_sha256": sha(OUT / "diagnostic_ocd/summary.json"), "trace_sha256": sha(trace_path), "diagnostic_only": True})
    print(json.dumps({k: v["summary"]["aggregate_commit_ct"] for k, v in streams.items()}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
