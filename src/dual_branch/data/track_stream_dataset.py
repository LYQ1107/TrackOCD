from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def stream_fname(stream):
    return "val_gt_track_stream.jsonl" if stream == "main" else f"val_gt_track_stream_{stream[5:]}.jsonl"


def load_stream_rows(stream, legacy_root=None):
    root = Path(legacy_root) if legacy_root else PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public"
    rows = []
    with open(root / stream_fname(stream)) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_all_val_embeddings():
    """DINO track-mean embeddings for all val GT tracks (A1 discovery space)."""
    from src.ocd_v2.common import load_mean_features
    return load_mean_features("dinov2", "gt_tracks_mean")
