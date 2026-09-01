"""Frozen CHP class split for Phase 4H.

The split is derived deterministically from the already-frozen Phase 3B
meta split (38 meta-train + 10 meta-dev classes) and is never re-partitioned
by official results:

  episode_pool: 38 meta-train classes (may play episode-known or
                 episode-pseudo-novel roles during CHP training);
  heldout:      10 meta-dev classes (never used in any CHP training episode;
                 used only for the real held-out proxy evaluation).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.orbit.protocol import meta_classes

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
SPLIT_PATH = ROOT / "configs" / "orbit_chp" / "class_split.json"


def load_chp_split(force=False) -> dict:
    """Return the frozen split dict, materializing it on first use."""
    if SPLIT_PATH.exists() and not force:
        return json.loads(SPLIT_PATH.read_text())
    meta_train = sorted(meta_classes("meta_train_classes"))
    meta_dev = sorted(meta_classes("meta_dev_classes"))
    split = {
        "episode_pool": meta_train,
        "heldout": meta_dev,
        "n_pool": len(meta_train),
        "n_heldout": len(meta_dev),
        "source": "outputs/orbit/splits/meta_train_classes.csv + meta_dev_classes.csv (frozen Phase 3B)",
        "policy": "heldout classes are NEVER used in CHP training episodes; "
                  "they are only used as real unseen pseudo-novel in proxy evaluation.",
    }
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split, indent=2))
    return split
