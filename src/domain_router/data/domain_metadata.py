from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load_train_domains():
    train = json.load(open(PROJECT_ROOT / "data/raw/tao/annotations/train.json"))
    vid2ds = {
        v["id"]: (v.get("metadata") or {}).get("dataset", "<none>")
        for v in train["videos"]
    }
    return vid2ds


def load_train_known_rows():
    rows = []
    with open(PROJECT_ROOT / "data/tao_ow_ocd_v1/public/train_known_tracks.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def domain_stats():
    vid2ds = load_train_domains()
    rows = load_train_known_rows()
    per = defaultdict(lambda: {"tracks": 0, "classes": set(), "videos": set()})
    for r in rows:
        ds = vid2ds.get(r["video_id"], "<none>")
        per[ds]["tracks"] += 1
        per[ds]["classes"].add(r["category_id"])
        per[ds]["videos"].add(r["video_id"])
    return {ds: {k: (len(v) if isinstance(v, set) else v) for k, v in d.items()}
            for ds, d in per.items()}
