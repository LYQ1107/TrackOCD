"""T8 fixed 64-episode learnability smoke on a balanced synthetic stream.

The synthetic vectors are deliberately linearly separable; this isolates
controller/state-machine correctness from the known DINOv2 geometry.  The
real-data frozen representation is evaluated only in the full experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase19r.data.episodes import MetaEpisode, StreamItem
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.training.rollout import rollout_batch


class SyntheticData:
    def __init__(self):
        self.supported_ids = list(range(48)); self.known_to_index = {i: i for i in range(48)}; self.active_known_mask = np.ones(48, bool); self.held_categories = set(); self.known_prototypes = np.zeros((48, 768), np.float32); self.known_bias = np.zeros(48, np.float32)
        for i in range(48): self.known_prototypes[i, i] = 1.0


def make_episodes(n: int = 64) -> list[MetaEpisode]:
    rng = np.random.default_rng(19); d = SyntheticData(); out = []
    for e in range(n):
        pseudo = [10, 11, 12]; visible = [0, 1, 2]; items = []
        # A fixed semantic skeleton contains all required positive/negative
        # actions; filler order is randomized without changing labels.
        spec = [(10, "pseudo_novel", "NEW", 1, False), (0, "visible_known", "KNOWN", 2, False),
                (11, "pseudo_novel", "NEW", 3, False), (10, "pseudo_novel", "EXISTING", 4, False),
                (12, "pseudo_novel", "NEW", 5, True), (1, "visible_known", "KNOWN", 6, False),
                (11, "pseudo_novel", "EXISTING", 7, False), (12, "pseudo_novel", "EXISTING", 8, False),
                (2, "visible_known", "KNOWN", 9, False), (None, "legal_unlabeled", "DEFER", 10, False)]
        for i in range(14):
            c = pseudo[i % 3] if i % 2 == 0 else visible[i % 3]; role = "pseudo_novel" if c in pseudo else "visible_known"; spec.append((c, role, "", 11 + i, False))
        for i, (cat, role, target, video, hard) in enumerate(spec):
            v = np.zeros(768, np.float32)
            if cat is not None: v[int(cat)] = 1.0
            if hard: v = .93 * v + .37 * np.eye(768, dtype=np.float32)[10]; v /= max(float(np.linalg.norm(v)), 1e-6)
            g = np.zeros(15, np.float32); g[0] = .9 if target != "DEFER" else -.7
            items.append(StreamItem(v, g, .9 if target != "DEFER" else .05, cat, role, f"e{e}:t{i}", video, 0, target, hard))
        mask = np.zeros(48, bool); mask[visible] = True
        out.append(MetaEpisode(tuple(visible), tuple(pseudo), mask, items, f"synthetic:{e}"))
    return out


def evaluate(model, data, episodes, device):
    model.eval(); rng = np.random.default_rng(99); total = good = neg = neg_good = known = known_good = existing = existing_good = 0
    with torch.no_grad():
        for ep in episodes:
            _, _, _, traces = rollout_batch(model, data, [ep], device, 0, 1, rng, train=False, ladder="L2", allow_defer=True)
            for rec in traces[0]:
                total += 1; good += int(rec["action"] == rec["target_kind"])
                if rec["target_kind"] == "NEW" and rec.get("state_count", 0) > 1: neg += 1; neg_good += int(rec["action"] == "NEW")
                if rec["target_kind"] == "KNOWN": known += 1; known_good += int(rec["action"] == "KNOWN")
                if rec["target_kind"] == "EXISTING": existing += 1; existing_good += int(rec["action"] == "EXISTING")
    return {"action_accuracy": good / total, "negative_new_with_memory_accuracy": neg_good / max(neg, 1), "known_role_accuracy": known_good / max(known, 1), "multi_state_existing_accuracy": existing_good / max(existing, 1), "counts": {"total": total, "negative_new_with_memory": neg, "known": known, "existing": existing}}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=800); p.add_argument("--device", default="cpu"); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    data = SyntheticData(); episodes = make_episodes(64); device = torch.device(a.device); model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias)).to(device); opt = torch.optim.AdamW(model.parameters(), lr=2e-3); rng = np.random.default_rng(1903)
    for _ in range(a.steps):
        opt.zero_grad(set_to_none=True); loss, _, _, _ = rollout_batch(model, data, episodes[:16], device, 1, 1, rng, train=True, ladder="L2", allow_defer=True, teacher_probability_override=1.0); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    m = evaluate(model, data, episodes, device); result = {"protocol": "trackocd_iclr27_phase19r_T8_small_overfit", "synthetic_linearly_separable": True, "steps": a.steps, "episodes": 64, "metrics": m, "thresholds": {"action_accuracy": .95, "negative_new_with_memory_accuracy": .95, "known_role_accuracy": .95, "multi_state_existing_accuracy": .90}, "passed": all([m["action_accuracy"] >= .95, m["negative_new_with_memory_accuracy"] >= .95, m["known_role_accuracy"] >= .95, m["multi_state_existing_accuracy"] >= .90])}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, indent=2, sort_keys=True));
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
