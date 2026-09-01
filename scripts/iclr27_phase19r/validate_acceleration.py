"""Semantic equivalence checks for the cached/indexed episode path."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase19r.data.episodes import EpisodeFactory, episode_to_index
from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.runtime.state import StateMemory
from src.iclr27_phase19r.training.rollout import _bundle_batch, rollout_batch


def _forward_first(model, data, episodes, device):
    memories = [StateMemory(max_states=model.max_states, max_anchors=8) for _ in episodes]
    items = [ep.items[0] for ep in episodes]
    bundles, _ = _bundle_batch(memories, [x.raw for x in items], items, device)
    raw = torch.from_numpy(np.stack([x.raw for x in items])).to(device)
    geom = torch.from_numpy(np.stack([x.geom for x in items])).to(device)
    q = torch.tensor([x.quality for x in items], dtype=torch.float32, device=device)
    km = torch.from_numpy(np.stack([ep.known_mask for ep in episodes])).to(device)
    return model(raw, geom, q, km, bundles, allow_defer=True)["logits"].detach().cpu()


def main() -> None:
    out = Path("outputs/iclr27_phase19r/audit/acceleration_equivalence.json")
    data = Phase19RData(0)
    pairs = []
    old_env = os.environ.get("PHASE19R_DISABLE_HARD_PAIR_CACHE")
    os.environ["PHASE19R_DISABLE_HARD_PAIR_CACHE"] = "1"
    old = EpisodeFactory(data, ladder="L2", validation=False)
    old_rng = np.random.default_rng(1902)
    for a in old.pseudo_pool[: min(6, len(old.pseudo_pool))]:
        for b in old.pseudo_pool[: min(6, len(old.pseudo_pool))]:
            if a != b:
                pairs.append((int(a), int(b), old._hard_pair(old_rng, int(a), int(b))))
    if old_env is None:
        os.environ.pop("PHASE19R_DISABLE_HARD_PAIR_CACHE", None)
    else:
        os.environ["PHASE19R_DISABLE_HARD_PAIR_CACHE"] = old_env
    new = EpisodeFactory(data, ladder="L2", validation=False)
    new_rng = np.random.default_rng(1902)
    pair_rows = []
    for a, b, expected in pairs:
        got = new._hard_pair(new_rng, a, b)
        pair_equal = expected[:2] == got[:2] and abs(float(expected[2]) - float(got[2])) <= 1e-6
        pair_rows.append({"source_category": a, "query_category": b,
                          "old": list(expected), "new": list(got), "equal": pair_equal,
                          "score_abs_diff": abs(float(expected[2]) - float(got[2]))})

    fixed_rng = np.random.default_rng(7719)
    fixed = [old.sample(fixed_rng) for _ in range(8)]
    idx_path = out.with_name("acceleration_fixed_index.jsonl")
    tmp = idx_path.with_name(idx_path.name + ".tmp")
    tmp.write_text("".join(json.dumps(episode_to_index(x), sort_keys=True) + "\n" for x in fixed)); os.replace(tmp, idx_path)
    indexed = EpisodeFactory(data, ladder="L2", validation=False, index_path=idx_path)
    replay = [indexed.sample(np.random.default_rng(0)) for _ in fixed]
    old_indices = [episode_to_index(x) for x in fixed]; new_indices = [episode_to_index(x) for x in replay]
    episode_equal = old_indices == new_indices

    torch.manual_seed(7788)
    m1 = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias))
    m2 = copy.deepcopy(m1)
    logits_diff = float(((_forward_first(m1, data, fixed, torch.device("cpu")) - _forward_first(m2, data, replay, torch.device("cpu"))).abs()).max())
    opt1 = torch.optim.AdamW(m1.parameters(), lr=3e-4); opt2 = torch.optim.AdamW(m2.parameters(), lr=3e-4)
    r1 = np.random.default_rng(8821); r2 = np.random.default_rng(8821)
    loss1, *_ = rollout_batch(m1, data, fixed, torch.device("cpu"), 1, 20, r1, train=True, ladder="L2", allow_defer=True)
    loss2, *_ = rollout_batch(m2, data, replay, torch.device("cpu"), 1, 20, r2, train=True, ladder="L2", allow_defer=True)
    opt1.zero_grad(); loss1.backward(); opt1.step(); opt2.zero_grad(); loss2.backward(); opt2.step()
    param_diff = max(float((a - b).abs().max()) for a, b in zip(m1.parameters(), m2.parameters()))
    result = {
        "protocol": "trackocd_iclr27_phase19r_acceleration_equivalence",
        "hard_pair_equal": bool(all(x["equal"] for x in pair_rows)),
        "hard_pair_rows": pair_rows,
        "episode_index_equal": bool(episode_equal),
        "controller_logits_max_abs_diff": logits_diff,
        "loss_old": float(loss1.detach()), "loss_indexed": float(loss2.detach()),
        "loss_abs_diff": float(abs(float(loss1.detach()) - float(loss2.detach()))),
        "one_optimizer_step_parameter_max_abs_diff": float(param_diff),
        "passed": bool(all(x["equal"] for x in pair_rows) and episode_equal and logits_diff <= 1e-6 and param_diff <= 1e-5),
        "physical_id_used_as_feature": False,
        "true_novel_labels": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(out.name + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); os.replace(tmp, out)
    print(json.dumps({k: result[k] for k in ("hard_pair_equal", "episode_index_equal", "controller_logits_max_abs_diff", "loss_abs_diff", "one_optimizer_step_parameter_max_abs_diff", "passed")}, sort_keys=True))


if __name__ == "__main__":
    main()
