"""Extract detailed Cross-Track Novel Reuse (CT-Reuse) cases from a strict
eval directory for the Phase 7A report."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-dir", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import csv
    summary = json.loads((ROOT / args.strict_dir / "summary.json").read_text())
    # recompute cross cases using the same definitions as strict_metrics
    from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
    from src.iclr27_phase4s.protocol import group_tracks, load_proposals
    from src.iclr27_phase7a.evaluation.strict_eval_any import load_gt_videos
    import numpy as np

    vids = summary.get("video_ids", [])
    all_rows = load_proposals(Path(args.proposals))
    tracks = group_tracks(all_rows)
    stream, labels = load_gt_videos(vids)
    mapping = align_pred_to_gt(tracks, gt_track_boxes(stream))
    aligned_keys = set(mapping)
    aligned = [r for r in all_rows
               if (int(r["video_id"]), int(r["track_id"])) in aligned_keys]
    first_seen = {}
    for a in aligned:
        sid = mapping[(int(a["video_id"]), int(a["track_id"]))]
        lab = labels[sid]
        if lab["protocol_role"] == "novel":
            first_seen.setdefault(int(lab["ground_truth_category_id"]), a)
    slot_cat = {}
    slot_birth_key = {}
    for a in aligned:
        sid = mapping[(int(a["video_id"]), int(a["track_id"]))]
        lab = labels[sid]
        if lab["protocol_role"] != "novel":
            continue
        act = a.get("sem_action")
        if act == "new":
            k = int(a["sem_sid"])
            slot_cat.setdefault(k, int(lab["ground_truth_category_id"]))
            slot_birth_key.setdefault(
                k, (int(a["video_id"]), int(a["track_id"])))
    cases = []
    for a in aligned:
        key = (int(a["video_id"]), int(a["track_id"]))
        sid = mapping[key]
        lab = labels[sid]
        if lab["protocol_role"] != "novel":
            continue
        cat = int(lab["ground_truth_category_id"])
        if first_seen.get(cat) is a:
            continue
        act = a.get("sem_action")
        k = int(a["sem_sid"]) if a.get("sem_sid") not in ("", None) else None
        if k is None:
            continue
        bk = slot_birth_key.get(k)
        if bk is None or bk == key:
            continue
        cases.append({
            "video_id": int(a["video_id"]),
            "frame_id": int(a["frame_id"]),
            "track_id": int(a["track_id"]),
            "category": cat,
            "action": act,
            "slot": k,
            "birth_track": list(bk),
            "correct": act == "existing" and slot_cat.get(k) == cat,
            "score": float(a["score"]),
        })
    cases.sort(key=lambda c: (c["video_id"], c["frame_id"]))
    out = {
        "n_cross_rows": len(cases),
        "n_correct": sum(1 for c in cases if c["correct"]),
        "cross_physical_reuse_acc": summary["strict"].get(
            "cross_physical_reuse_acc"),
        "cases": cases,
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "cases"},
                     indent=2))


if __name__ == "__main__":
    main()
