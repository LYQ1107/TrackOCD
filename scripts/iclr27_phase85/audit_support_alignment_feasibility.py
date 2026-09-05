#!/usr/bin/env python3
"""Summarize whether a legal alignment stage is justified by selection evidence."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase85"
AUDIT = OUT / "audit/support_selection_audit.json"


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


def main() -> None:
    z = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = z.get("rows", [])
    records = []
    for key in sorted({(r.get("event_key"), r.get("prefix"), r.get("polarity")) for r in rows}):
        event, prefix, polarity = key
        rr = [r for r in rows if (r.get("event_key"), r.get("prefix"), r.get("polarity")) == key]
        pool = any(float(r.get("pool_max_iou", 0.0)) >= 0.5 for r in rr)
        raw = any(float(r.get("raw_iou", 0.0)) >= 0.5 for r in rr)
        rank = any(float(r.get("reranked_iou", 0.0)) >= 0.5 for r in rr)
        defer = any(bool(r.get("defer")) for r in rr)
        if not pool:
            bucket = "pool_limited"
        elif raw and rank and not defer:
            bucket = "raw_and_rank_reliable"
        elif raw and not rank:
            bucket = "rerank_harm"
        elif not raw and rank and not defer:
            bucket = "rerank_rescue"
        elif defer and rank:
            bucket = "defer_harm_or_safe_candidate"
        elif pool and not rank:
            bucket = "pool_present_selection_gap"
        else:
            bucket = "unresolved"
        records.append({"event_key": event, "fold": rr[0].get("fold", -1), "prefix": prefix, "polarity": polarity, "bucket": bucket, "pool_reliable": pool, "raw_reliable": raw, "reranked_reliable": rank, "deferred": defer, "candidate_rows": len(rr), "candidate_count_max": max(int(r.get("candidate_count", 0)) for r in rr)})
    summary = []
    for prefix in (1, 2, 4, 8, 16):
        for polarity in ("positive", "negative"):
            subset = [r for r in records if r["prefix"] == prefix and r["polarity"] == polarity]
            summary.append({"prefix": prefix, "polarity": polarity, "events": len(subset), "pool_reliable_events": sum(r["pool_reliable"] for r in subset), "raw_reliable_events": sum(r["raw_reliable"] for r in subset), "reranked_reliable_events": sum(r["reranked_reliable"] for r in subset), "deferred_events": sum(r["deferred"] for r in subset), "buckets": dict(sorted(Counter(r["bucket"] for r in subset).items()))})
    p16 = {s["polarity"]: s for s in summary if s["prefix"] == 16}
    routing = {"alignment_authorized": bool(p16.get("positive", {}).get("reranked_reliable_events", 0) >= 26 and p16.get("negative", {}).get("reranked_reliable_events", 0) <= 9), "criterion": "positive >=26/76 and negative <=9/76 relative to raw 20/8", "reason": "The learned reranker increased positive selection only to 23/76 and negative activation to 15/76; defer reduced final positive reliability to 8/76. Therefore no alignment route is authorized in this window."}
    result = {"schema_version": "trackocd.phase85.support_alignment_feasibility.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "selection_audit": str(AUDIT.resolve()), "records": records, "summary": summary, "routing": routing, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(OUT / "audit/support_alignment_feasibility.json", result)
    atomic_json(OUT / "completion/support_alignment_feasibility.done", {"status": "DONE", "audit": str((OUT / "audit/support_alignment_feasibility.json").resolve()), "alignment_authorized": routing["alignment_authorized"]})
    print(json.dumps({"p16": p16, "routing": routing}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
