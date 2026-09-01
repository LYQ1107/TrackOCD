"""Shared model-in-the-loop rollout for ADSSI (pilot + dev)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from src.iclr27_phase4y.model import DynamicStateMemory


def run_track(model, tsr, mem, z_seq, q_seq, min_age, commit_threshold,
              margin_ratio, device):
    """Returns (commit, commit_t). z_seq: (T,256), q_seq: (T,6)."""
    state = tsr.init_state(1, device)
    last_s = None
    for t in range(len(z_seq)):
        f = torch.from_numpy(z_seq[t:t + 1]).to(device)
        qt = torch.from_numpy(q_seq[t:t + 1]).to(device)
        s, state = tsr.step(f, qt, state)
        if t < min_age:
            continue
        zt = model.obs(s)
        last_s = s
        scores, prop, _, _ = mem.infer(zt, float(q_seq[t, 0]))
        post = torch.softmax(scores, dim=-1)
        top2 = torch.topk(post, k=2).values
        if float(post.max()) < commit_threshold:
            continue
        if float(top2[0] / max(top2[1], 1e-9)) < margin_ratio:
            continue
        a = int(post.argmax())
        return a, t, scores, prop, last_s
    return None, None, None, None, last_s


def run_episode_eval(model, ep, store, cfg, tsr, anchors, cat_index, device,
                     commit_threshold=0.5, margin_ratio=1.5, min_age=2):
    active_idx = [cat_index[c] for c in ep["pseudo_known"]]
    mem = DynamicStateMemory(model, anchors[active_idx], device)
    slot_cat = {}
    stats = defaultdict(int)
    outcomes = []
    for occ in ep["occurrences"]:
        z, q = store.tracklet_seq(occ["key"])
        n = min(len(z), cfg.max_len)
        a, t, scores, prop, s_commit = run_track(
            model, tsr, mem, z[:n], q[:n], min_age, commit_threshold,
            margin_ratio, device)
        commit = None
        if a is not None:
            C = len(active_idx)
            if a < C:
                commit = ("known", ep["pseudo_known"][a], t)
            elif a < C + mem.size():
                k = a - C
                commit = ("existing", k, t)
                mem.update(k, model.obs(s_commit),
                           float(np.clip(q[t, 0], 0.05, 0.95)))
            elif a == C + mem.size():
                k = mem.create(prop, float(np.clip(q[t, 0], 0.05, 0.95)))
                commit = ("new", k, t)
                slot_cat[occ["category"]] = k
            else:
                commit = ("noise", None, t)
        outcomes.append((occ, commit))
    for occ, commit in outcomes:
        cat = occ["category"]
        if occ["role"] == "known":
            stats["known_total"] += 1
            if commit is not None and commit[0] == "known" and commit[1] == cat:
                stats["known_correct"] += 1
            elif commit is not None and commit[0] == "new":
                stats["known_to_new"] += 1
            elif commit is not None and commit[0] == "existing":
                stats["known_to_existing"] += 1
            elif commit is not None:
                stats["known_to_noise"] += 1
            else:
                stats["known_unresolved"] += 1
        elif occ["role"] == "novel":
            if occ.get("first"):
                stats["first_total"] += 1
                if commit is not None and commit[0] == "new":
                    stats["first_correct"] += 1
                elif commit is not None and commit[0] == "known":
                    stats["absorbed"] += 1
                elif commit is not None and commit[0] == "existing":
                    stats["wrong_reuse"] += 1
                elif commit is not None:
                    stats["first_to_noise"] += 1
                else:
                    stats["first_unresolved"] += 1
            else:
                stats["later_total"] += 1
                if (commit is not None and commit[0] == "existing"
                        and commit[1] == slot_cat.get(cat, -1)):
                    stats["reuse_correct"] += 1
                elif commit is not None and commit[0] == "new":
                    stats["overbirth"] += 1
                elif commit is not None and commit[0] == "known":
                    stats["absorbed"] += 1
                elif commit is not None and commit[0] == "existing":
                    stats["wrong_reuse"] += 1
                elif commit is not None:
                    stats["later_to_noise"] += 1
                else:
                    stats["later_unresolved"] += 1
        else:
            if commit is not None and commit[0] == "new":
                stats["fp_born"] += 1
            elif commit is not None and commit[0] == "noise":
                stats["fp_no_write"] += 1
            elif commit is not None:
                stats["fp_other_commit"] += 1
            else:
                stats["fp_unresolved"] += 1
    return stats, mem.size()
