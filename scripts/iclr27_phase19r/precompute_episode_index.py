"""Precompute reproducible, feature-free causal episode index shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.iclr27_phase19r.data.episodes import EpisodeFactory, episode_to_index
from src.iclr27_phase19r.data.stream import Phase19RData


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(4), default=0)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--seed", type=int, default=1902)
    p.add_argument("--ladder", choices=["L0", "L1", "L2"], default="L2")
    p.add_argument("--validation", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = Phase19RData(a.fold, final=a.final)
    factory = EpisodeFactory(data, ladder=a.ladder, validation=a.validation)
    rng = np.random.default_rng(int(a.seed) * 1009 + int(a.fold) + (17 if a.validation else 0))
    meta = {
        "protocol": "trackocd_iclr27_phase19r_episode_index_v1",
        "fold": int(a.fold), "final": bool(a.final), "validation": bool(a.validation),
        "ladder": str(a.ladder), "episodes": int(a.episodes), "seed": int(a.seed),
        "feature_source": "referenced by Phase19RData; no feature arrays serialized",
        "target_semantics": "EpisodeFactory._assign_oracle_targets",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    with tmp.open("w") as fh:
        for _ in range(int(a.episodes)):
            fh.write(json.dumps(episode_to_index(factory.sample(rng)), sort_keys=True) + "\n")
    os.replace(tmp, a.out)
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    manifest = dict(meta, sha256=digest, bytes=a.out.stat().st_size,
                    source_rows=int(data.summary()["source_rows"]),
                    model_facing_category_values="opaque role/mask only; category is loss-side metadata")
    mpath = a.out.with_name(a.out.name + ".json")
    mtmp = mpath.with_name(mpath.name + ".tmp"); mtmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n"); os.replace(mtmp, mpath)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
