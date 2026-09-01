"""Stream-level physical + semantic statistics for any proposals CSV.

GT alignment is intentionally not performed here (the Q1 dev alignment is
done by strict_eval_any / the frozen Phase 5A evaluator); this script only
reports causal-track statistics and semantic-memory statistics that do not
need private labels.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase4s.protocol import load_proposals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load_proposals(Path(args.csv))
    n_rows = len(rows)
    tracks = defaultdict(list)
    for i, r in enumerate(rows):
        tracks[(int(r["video_id"]), int(r["track_id"]))].append(i)
    lens = np.asarray([len(v) for v in tracks.values()])
    n_tracks = len(lens)
    actions = defaultdict(int)
    novel_slots = set()
    slot_tracks = defaultdict(set)
    n_new = 0
    for r in rows:
        a = r.get("sem_action") or ""
        actions[a] += 1
        if a == "new":
            n_new += 1
        sid = r.get("sem_sid")
        if sid not in ("", None) and a in ("new", "existing"):
            novel_slots.add(int(sid))
            slot_tracks[int(sid)].add(
                (int(r["video_id"]), int(r["track_id"])))
    cross_slots = [sid for sid, trs in slot_tracks.items() if len(trs) > 1]
    first_scores = {}
    for r in rows:
        key = (int(r["video_id"]), int(r["track_id"]))
        first_scores.setdefault(key, float(r["score"]))
    fs = np.asarray(list(first_scores.values()))
    result = {
        "n_rows": n_rows,
        "n_tracks": n_tracks,
        "n_frames": len({(int(r["video_id"]), int(r["frame_id"]))
                         for r in rows}),
        "track_len1_frac": float(np.mean(lens == 1)) if n_tracks else 0.0,
        "median_track_len": float(np.median(lens)) if n_tracks else 0.0,
        "first_score_mean": float(fs.mean()) if len(fs) else 0.0,
        "semantic_actions": dict(actions),
        "n_new_actions": n_new,
        "n_novel_slots": len(novel_slots),
        "n_cross_physical_slots": len(cross_slots),
        "cross_physical_slots": sorted(cross_slots),
        "max_slot_evidence": 0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
