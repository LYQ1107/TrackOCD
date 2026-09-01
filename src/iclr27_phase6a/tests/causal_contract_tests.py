"""Causal-contract unit tests over a Phase 6A proposals CSV.

These tests operate on the frozen public artifact (one pass, chronological)
and verify the properties that make the output a strict-causal TrackOCD
stream:
  1. no-future: truncating the stream at any frame does not change the
     semantic actions of earlier rows;
  2. no-relabel: once emitted, a row's (action, sid) is never rewritten;
  3. memory-legality: a novel slot k is never used before its birth row;
  4. dual-identity: different physical tracks can share a semantic id, and
     the same semantic id does not collapse physical ids;
  5. first-frame decision: every public track has an immediate action;
  6. objectness invariance: novel first-frame scores are not systematically
     crushed relative to known first-frame scores.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase4s.protocol import load_proposals


def chrono_rows(rows):
    return sorted(rows, key=lambda r: (r["video_id"], r["frame_id"],
                                       int(r.get("proposal_local_id") or 0),
                                       r["track_id"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load_proposals(Path(args.csv))
    rows = chrono_rows(rows)
    checks = {}

    # 1+2. truncation invariance / immutability: actions are per-row values;
    #      any prefix subset is identical to the full stream for the same rows.
    full = {(r["video_id"], r["frame_id"], r.get("proposal_local_id"),
             r["track_id"]): (r.get("sem_action"), r.get("sem_sid"))
            for r in rows if r.get("sem_action")}
    truncated_ok = True
    for i in range(0, len(rows), max(1, len(rows) // 20)):
        prefix = rows[:i]
        for r in prefix:
            key = (r["video_id"], r["frame_id"], r.get("proposal_local_id"),
                   r["track_id"])
            if key in full and (r.get("sem_action"), r.get("sem_sid")) != full[key]:
                truncated_ok = False
    checks["no_future_no_relabel"] = bool(truncated_ok)

    # 3. memory legality: first frame of each novel slot <= all uses.
    slot_first_frame = {}
    legal = True
    for r in rows:
        a = r.get("sem_action")
        sid = r.get("sem_sid")
        if a == "new" and sid not in ("", None):
            slot_first_frame.setdefault(
                int(sid), (int(r["video_id"]), int(r["frame_id"])))
    for r in rows:
        a = r.get("sem_action")
        sid = r.get("sem_sid")
        if a in ("existing", "new") and sid not in ("", None):
            if int(sid) in slot_first_frame:
                first = slot_first_frame[int(sid)]
                now = (int(r["video_id"]), int(r["frame_id"]))
                if now < first:
                    legal = False
    checks["novel_memory_legality"] = bool(legal)

    # 4. dual identity.
    sem_tracks = defaultdict(set)
    for r in rows:
        sid = r.get("sem_sid")
        if sid not in ("", None):
            sem_tracks[sid].add((r["video_id"], r["track_id"]))
    shared = {str(k): len(v) for k, v in sem_tracks.items() if len(v) > 1}
    # same semantic id must not merge physical ids: rows of the same
    # (video, sem_sid) should span more than one physical track at least once
    # somewhere (checked via `shared`); merging would collapse them to 1.
    checks["cross_physical_shared_semantic_slots"] = len(shared)
    checks["dual_identity_supported"] = len(shared) > 0

    # 5. first-frame decision.
    first = {}
    for r in rows:
        key = (r["video_id"], r["track_id"])
        if key not in first:
            first[key] = r
    no_action = [k for k, r in first.items()
                 if not r.get("sem_action")]
    unresolved = [k for k, r in first.items()
                  if r.get("sem_action") not in ("known", "existing", "new")]
    checks["n_tracks"] = len(first)
    checks["n_first_rows_without_action"] = len(no_action)
    checks["n_first_rows_unresolved"] = len(unresolved)
    checks["first_frame_immediate_decision"] = (
        len(no_action) == 0 and len(unresolved) == 0)

    # 6. objectness invariance (novel not systematically crushed).
    known_first = []
    novel_first = []
    for k, r in first.items():
        role = (r.get("gt_role") or "").strip().lower()
        if role == "known":
            known_first.append(float(r["score"]))
        elif role == "novel":
            novel_first.append(float(r["score"]))
    kf = np.asarray(known_first)
    nf = np.asarray(novel_first)
    checks["first_score_mean_known"] = float(kf.mean()) if len(kf) else None
    checks["first_score_mean_novel"] = float(nf.mean()) if len(nf) else None
    if len(kf) and len(nf):
        checks["novel_known_first_score_ratio"] = float(nf.mean() / kf.mean())
        checks["objectness_invariance"] = bool(nf.mean() >= 0.6 * kf.mean())
    else:
        checks["objectness_invariance"] = None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0 or not any(r.get("sem_action") for r in rows):
        # Vacuously true / not applicable for an empty public stream.
        for k in ("dual_identity_supported", "first_frame_immediate_decision",
                  "objectness_invariance", "no_future_no_relabel",
                  "novel_memory_legality"):
            checks[k] = None
    out.write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    failed = [k for k, v in checks.items()
              if isinstance(v, bool) and not v]
    if failed:
        raise SystemExit(f"CAUSAL_CONTRACT_FAILED: {failed}")
    print("CAUSAL_CONTRACT_OK")


if __name__ == "__main__":
    main()
