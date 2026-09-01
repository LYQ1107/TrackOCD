"""Small cProfile pass used to choose the next speed repair."""
from __future__ import annotations

import cProfile
import json
import pstats
import io
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase19r.data.episodes import EpisodeFactory
from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.training.rollout import rollout_batch


def main() -> None:
    data = Phase19RData(0); f = EpisodeFactory(data, ladder="L2", validation=False)
    eps = [f.sample(np.random.default_rng(1902 + i)) for i in range(24)]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias)).to(dev)
    rng = np.random.default_rng(9917)
    pr = cProfile.Profile(); pr.enable()
    for u in range(20):
        loss, _, _, _ = rollout_batch(model, data, eps, dev, u + 1, 20, rng, train=True, ladder="L2", allow_defer=True)
        loss.backward(); model.zero_grad(set_to_none=True)
    pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(35)
    out = Path("outputs/iclr27_phase19r/audit/rollout_profile.txt"); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(s.getvalue())
    print(s.getvalue())


if __name__ == "__main__": main()
