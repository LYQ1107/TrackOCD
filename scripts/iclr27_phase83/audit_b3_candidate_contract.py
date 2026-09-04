#!/usr/bin/env python3
"""Audit B2/B3 public-row candidate grouping against native Q0 candidates."""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/iclr27_phase83"
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); public = defaultdict(int)
    for r in rows: public[(int(r["video_id"]), int(r["image_id"]))] += 1
    native = defaultdict(int)
    with NATIVE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("bbox_xyxy") is not None: native[(int(r["video_id"]), int(r["image_id"]))] += 1
    details = []; mismatch = Counter(); by_fold = defaultdict(lambda: {"rows": 0, "public_native_exact": 0, "q0_native_exact": 0, "public_q0_exact": 0})
    for line in OBS.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        e = json.loads(line)
        for side in ("source", "target"):
            for d in e.get(side + "_row_details", []):
                key = (int(d.get("video_id", -1)), int(d.get("image_id", -1))); pc = public.get(key, 0); nc = native.get(key, 0); qc = int(d.get("q0_candidate_count", 0) or 0); item = {"event_key": e.get("event_key"), "model_event_uid": e.get("model_event_uid"), "fold": int(e.get("fold", -1)), "prefix": int(e.get("prefix", 0)), "polarity": e.get("polarity"), "side": side, "video_id": key[0], "image_id": key[1], "public_row_candidate_count": pc, "native_q0_candidate_count": nc, "observability_q0_candidate_count": qc, "public_minus_native": pc - nc, "q0_minus_native": qc - nc, "public_minus_q0": pc - qc, "q0_reliable": bool(d.get("q0_reliable", False))}; details.append(item); mismatch[("public_native_exact" if pc == nc else "public_native_mismatch")] += 1; mismatch[("q0_native_exact" if qc == nc else "q0_native_mismatch")] += 1; mismatch[("public_q0_exact" if pc == qc else "public_q0_mismatch")] += 1; b = by_fold[str(item["fold"])]
                b["rows"] += 1; b["public_native_exact"] += int(pc == nc); b["q0_native_exact"] += int(qc == nc); b["public_q0_exact"] += int(pc == qc)
    diffs = [x["public_minus_native"] for x in details]; qdiffs = [x["q0_minus_native"] for x in details]; summary = {"rows": len(details), "public_native_exact_rate": mismatch["public_native_exact"] / max(len(details), 1), "q0_native_exact_rate": mismatch["q0_native_exact"] / max(len(details), 1), "public_q0_exact_rate": mismatch["public_q0_exact"] / max(len(details), 1), "public_minus_native_mean": float(sum(diffs) / max(len(diffs), 1)), "q0_minus_native_mean": float(sum(qdiffs) / max(len(qdiffs), 1)), "public_minus_native_abs_median": float(sorted(abs(x) for x in diffs)[len(diffs) // 2]) if diffs else 0.0, "q0_minus_native_abs_median": float(sorted(abs(x) for x in qdiffs)[len(qdiffs) // 2]) if qdiffs else 0.0, "mismatch_counts": dict(mismatch), "by_fold": dict(by_fold), "interpretation": "B2/B3 public-row grouping is contract-aligned only when public and native counts agree; observability q0 count is the runtime reference.", "public_dev_q1_sealed_accessed": False}
    out = {"schema_version": "trackocd.phase83.b3.candidate_contract_audit.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "summary": summary, "details": details, "public_csv": str(CSV_PATH.resolve()), "native_lineage": str(NATIVE.resolve()), "observability": str(OBS.resolve()), "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(OUT / "audit/b3_candidate_contract_audit.json", out); atomic_json(OUT / "status.json", {"phase": "Phase83", "route": "B3_CANDIDATE_CONTRACT_AUDIT", "status": "COMPLETE", "summary": summary, "public_dev_q1_sealed_accessed": False}); atomic_json(OUT / "completion/b3_candidate_contract_audit.done", {"status": "DONE", "metrics": str((OUT / "audit/b3_candidate_contract_audit.json").resolve())}); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
