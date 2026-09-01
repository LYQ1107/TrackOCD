"""Generate the preregistered 10k mixed-episode distribution audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.data.episodes import EpisodeFactory, episode_distribution


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--fold", type=int, default=0); p.add_argument("--count", type=int, default=10000); p.add_argument("--ladder", choices=["L0", "L1", "L2"], default="L2"); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); data = Phase19RData(a.fold); factory = EpisodeFactory(data, ladder=a.ladder, validation=True)
    audit = episode_distribution(factory, np.random.default_rng(1902 + a.fold), a.count)
    audit.update({"fold": a.fold, "ladder": a.ladder, "validation": True,
                  "pseudo_pool": len(factory.pseudo_pool), "visible_pool": len(factory.visible_pool)})
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_name(a.out.name + ".tmp"); tmp.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n"); tmp.replace(a.out)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__": main()
