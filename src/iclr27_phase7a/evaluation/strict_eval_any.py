"""Strict-causal TrackOCD evaluation for an arbitrary video set.

Identical metric definitions to Phase 5A/6C strict_causal_eval, but the GT
stream can be restricted to any video list (used for Q1 DEV and the locked
heldout split). GT is used only for evaluation, never for decisions.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import (
    group_tracks,
    load_proposals,
)
from src.iclr27_phase5a.evaluation.strict_causal_eval import (
    strict_metrics,
)
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def load_gt_videos(video_ids):
    vids = set(int(v) for v in video_ids)
    stream = []
    with open(ROOT / "data/trackocd_v1/pure/public/val_gt_track_stream.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if int(r["video_id"]) in vids:
                    stream.append(r)
    labels_all = {}
    with open(ROOT / "data/trackocd_v1/pure/private/val_gt_track_labels.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels_all[r["sample_id"]] = r
    labels = {r["sample_id"]: labels_all[r["sample_id"]] for r in stream
              if r["sample_id"] in labels_all}
    stream = [r for r in stream if r["sample_id"] in labels]
    return stream, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--video-ids", required=True,
                    help="JSON list of TAO video ids")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_rows = load_proposals(Path(args.proposals))
    arr = np.load(ROOT / args.feats)["feats"]
    assert len(arr) == len(all_rows)
    rows = all_rows
    tracks = group_tracks(rows)
    video_ids = json.loads(args.video_ids)
    stream, labels = load_gt_videos(video_ids)
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)
    aligned_keys = set(mapping)
    aligned_rows = [r for r in rows if (int(r["video_id"]), int(r["track_id"]))
                    in aligned_keys]
    records_by_row = {}
    for r in rows:
        a = r.get("sem_action") or "unresolved"
        sid = int(r["sem_sid"]) if r.get("sem_sid") not in ("", None) else None
        key = (int(r["video_id"]), int(r["track_id"]))
        records_by_row[id(r)] = {
            "key": list(key), "frame_id": int(r["frame_id"]),
            "row_id": id(r), "action": a, "sid": sid,
            "age": 1.0,
        }
    n_born_global = sum(1 for r in rows if r.get("sem_action") == "new")
    sm = strict_metrics(records_by_row, aligned_rows, labels, mapping,
                        n_born_global=n_born_global)

    # legacy per-track first/last predictions
    first_preds, last_preds = [], []
    order = 0
    by_track_records = defaultdict(list)
    for r in rows:
        a = r.get("sem_action")
        sid = int(r["sem_sid"]) if r.get("sem_sid") not in ("", None) else None
        key = (int(r["video_id"]), int(r["track_id"]))
        by_track_records[key].append({
            "frame_id": int(r["frame_id"]), "action": a, "sid": sid})
    for key, sid in sorted(mapping.items()):
        order += 1
        recs = sorted(by_track_records.get(key, []),
                      key=lambda r: r["frame_id"])
        if not recs:
            first_preds.append({"sample_id": sid, "prediction_type": "unresolved",
                                "stream_order": order})
            last_preds.append({"sample_id": sid, "prediction_type": "unresolved",
                               "stream_order": order})
            continue
        for target, out in ((recs[0], first_preds), (recs[-1], last_preds)):
            if target["action"] == "known":
                out.append({"sample_id": sid, "prediction_type": "known",
                            "semantic_category_id": target["sid"],
                            "stream_order": order})
            elif target["action"] in ("existing", "new"):
                out.append({"sample_id": sid, "prediction_type": "novel",
                            "virtual_category_id": target["sid"],
                            "stream_order": order})
            else:
                out.append({"sample_id": sid, "prediction_type": "unresolved",
                            "stream_order": order})
    ev = TrackOCDEvaluator([labels[sid] for sid in sorted(labels)])
    legacy_first = ev.evaluate(first_preds)
    legacy_last = ev.evaluate(last_preds)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "video_ids": video_ids,
        "strict": sm,
        "legacy_first_frame": {k: (float(v) if isinstance(v, (int, float))
                                   else v) for k, v in legacy_first.items()
                               if k != "hungarian_assignment"},
        "legacy_last_frame": {k: (float(v) if isinstance(v, (int, float))
                                  else v) for k, v in legacy_last.items()
                              if k != "hungarian_assignment"},
        "n_rows": len(rows),
        "n_aligned_tracks": len(mapping),
        "n_gt_tracks": len(labels),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 default=float))
    print(json.dumps(summary["strict"], indent=2, default=float))


if __name__ == "__main__":
    main()
