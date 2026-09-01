#!/usr/bin/env python
"""Precompute feature-free event-aligned episode shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.iclr27_phase19r.data.stream import Phase19RData, OUT
from src.iclr27_phase19r.training.event_aligned import event_index_spec, load_event_manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(4), required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--seed", type=int, default=1902)
    p.add_argument("--events", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = Phase19RData(a.fold)
    event_path = a.events or (OUT / "manifests" / f"pseudo_train_events_fold{a.fold}.jsonl")
    events = load_event_manifest(event_path)
    if not events:
        raise ValueError(f"empty pseudo event manifest: {event_path}")
    rng = np.random.default_rng(int(a.seed) * 1013 + int(a.fold) + 7919)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    with tmp.open("w") as fh:
        for i in range(int(a.episodes)):
            event = events[i % len(events)]
            spec = event_index_spec(data, event, rng=rng,
                                    episode_id=f"f{a.fold}:event_aligned:{i:08d}:src{event['event_key']}")
            fh.write(json.dumps(spec, sort_keys=True) + "\n")
    os.replace(tmp, a.out)
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    meta = {"protocol": "trackocd_iclr27_phase19r_event_aligned_index_v1",
            "fold": int(a.fold), "episodes": int(a.episodes), "seed": int(a.seed),
            "event_manifest": str(event_path), "event_manifest_events": len(events),
            "sha256": digest, "bytes": a.out.stat().st_size,
            "feature_source": "referenced by Phase19RData; no feature arrays serialized",
            "model_facing_category_values": "opaque role/mask only; category is loss-side metadata",
            "masked_categories_source": "supported TRAIN categories only"}
    mp = a.out.with_name(a.out.name + ".json"); mt = mp.with_name(mp.name + ".tmp")
    mt.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n"); os.replace(mt, mp)
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

