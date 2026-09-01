from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def stream_fname(stream):
    return "val_gt_track_stream.jsonl" if stream == "main" else f"val_gt_track_stream_{stream[5:]}.jsonl"


def load_video_streams(stream, frames_feats=None):
    """Group GT tracks by video, ordered by first frame (causal order)."""
    rows = []
    with open(PROJECT_ROOT / "data/tao_ow_ocd_v1/public" / stream_fname(stream)) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    order = {}
    for vid, vrows in by_video.items():
        vrows.sort(key=lambda r: min(r.get("frame_ids", [0])))
        order[vid] = vrows
    return order
