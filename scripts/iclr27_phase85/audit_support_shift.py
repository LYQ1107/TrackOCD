#!/usr/bin/env python3
"""Compare legal TRAIN support groups with frozen event candidate conditions."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase85"
MANIFEST = OUT / "manifests/phase85_support_prefix_manifest.json"
REPLAY = OUT / "metrics/support_event_replay.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("min", "p10", "median", "p90", "max", "mean")}
    x = np.asarray(values, np.float64)
    return {"min": float(x.min()), "p10": float(np.quantile(x, .1)), "median": float(np.median(x)), "p90": float(np.quantile(x, .9)), "max": float(x.max()), "mean": float(x.mean())}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    data_z = np.load(manifest["data"], allow_pickle=False)
    offsets = np.asarray(data_z["offsets"], np.int64)
    train = defaultdict(lambda: {"candidate_count": [], "support_quality": [], "source_variance": [], "source_length": [], "match": 0, "defer": 0, "groups": 0})
    for gi, meta in enumerate(manifest.get("groups_meta", [])):
        p = str(meta["prefix"]); d = train[p]; d["groups"] += 1
        d["candidate_count"].append(float(offsets[gi + 1] - offsets[gi]) if gi + 1 < len(offsets) else 0.0)
        d["support_quality"].append(float(meta.get("support_quality", 0.0)))
        d["source_variance"].append(float(meta.get("source_variance", 0.0)))
        d["source_length"].append(float(meta.get("source_length", 0)))
        d["match" if meta.get("kind") == "positive" else "defer"] += 1
    event = defaultdict(lambda: {"candidate_count": [], "topk_count": [], "raw_margin": [], "defer_probability": [], "raw_reliable": 0, "rerank_reliable": 0, "final_reliable": 0, "rows": 0})
    for record in replay.get("records", []):
        p = str(record.get("prefix")); d = event[(p, str(record.get("polarity")))]
        for row in record.get("target_rows", []):
            d["rows"] += 1
            d["candidate_count"].append(float(row.get("candidate_count", 0)))
            d["topk_count"].append(float(row.get("topk_count", 0)))
            d["defer_probability"].append(float(row.get("defer_probability", 0.0)))
            if row.get("raw_reliable"): d["raw_reliable"] += 1
            if row.get("reranked_reliable"): d["rerank_reliable"] += 1
            if row.get("final_reliable"): d["final_reliable"] += 1
    summary = {"train": {}, "event": {}}
    for p, d in train.items():
        summary["train"][p] = {"groups": d["groups"], "match_groups": d["match"], "defer_groups": d["defer"], "candidate_count": quantiles(d["candidate_count"]), "support_quality": quantiles(d["support_quality"]), "source_variance": quantiles(d["source_variance"]), "source_length": quantiles(d["source_length"])}
    for key, d in event.items():
        p, polarity = key
        summary["event"].setdefault(p, {})[polarity] = {"rows": d["rows"], "candidate_count": quantiles(d["candidate_count"]), "topk_count": quantiles(d["topk_count"]), "defer_probability": quantiles(d["defer_probability"]), "raw_reliable_rows": d["raw_reliable"], "rerank_reliable_rows": d["rerank_reliable"], "final_reliable_rows": d["final_reliable"]}
    result = {"schema_version": "trackocd.phase85.support_shift_audit.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "manifest": str(MANIFEST.resolve()), "replay": str(REPLAY.resolve()), "summary": summary, "interpretation": ["TRAIN groups are prefix-expanded legal supervision; event rows are frozen causal target rows and are not used for fitting.", "Differences in candidate-count and defer distributions are diagnostic evidence only; no threshold or checkpoint is selected from held events."], "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(OUT / "audit/support_train_event_shift.json", result)
    atomic_json(OUT / "completion/support_train_event_shift.done", {"status": "DONE", "audit": str((OUT / "audit/support_train_event_shift.json").resolve())})
    print(json.dumps({"train": summary["train"], "event_p16": summary["event"].get("16", {})}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
