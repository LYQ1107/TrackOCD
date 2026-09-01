"""Baselines B0/B1/B2 over the pseudo-novel episodic benchmark."""
from __future__ import annotations

import numpy as np
import torch

from src.iclr27_phase4s.model import NovelMemory, SemanticCore
from src.iclr27_phase4s.runtime import model_action_kind, teacher_targets_for_episode


def b0_episode(batch, cfg, raw_protos, tau_known, tau_novel):
    """Frozen assign-or-create on raw DINO features. Immediate decision at the
    first frame of each occurrence; own-action-driven memory.
    Returns (outcomes, n_slots)."""
    proto_cats = sorted(batch["pseudo_known"])
    P = np.stack([raw_protos[c] for c in proto_cats])
    n_occ = batch["feats"].shape[1]
    memory = []  # (proto 768, provenance cat)
    outcomes = []
    for i in range(n_occ):
        role, first = batch["role_first"][i]
        if int(batch["mask"][i].sum()) == 0:
            outcomes.append(None)
            continue
        z = batch["feats"][i, 0].cpu().numpy().astype(np.float32)
        z = z / (np.linalg.norm(z) + 1e-12)
        sim_k = P @ z
        j = int(sim_k.argmax())
        if sim_k[j] >= tau_known:
            outcomes.append(("known", proto_cats[j], 0))
        else:
            sims = [float(m[0] @ z) for m in memory]
            if sims and max(sims) >= tau_novel:
                best = int(np.argmax(sims))
                outcomes.append(("existing", memory[best][1], 0))
            else:
                outcomes.append(("new", int(batch["cats"][i]), 0))
                memory.append((z, int(batch["cats"][i])))
    return outcomes, len(memory)


def b0_episode_track(rows, feats_by_key, raw_protos, tau_known, tau_novel, mem):
    """Frozen assign-or-create for ONE dev physical track. First observation
    decides; `mem` is a persistent list of raw prototypes (virtual novel slots)."""
    proto_cats = sorted(raw_protos)
    P = np.stack([raw_protos[c] for c in proto_cats])
    z = None
    for r in rows:
        f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]), int(r["image_id"])))
        if f is not None:
            z = f.astype(np.float32)
            break
    if z is None:
        return ("unresolved", None, -1)
    z = z / (np.linalg.norm(z) + 1e-12)
    sim_k = P @ z
    j = int(sim_k.argmax())
    if sim_k[j] >= tau_known:
        return ("known", proto_cats[j], 0)
    sims = [float(m @ z) for m in mem]
    if sims and max(sims) >= tau_novel:
        k = int(np.argmax(sims))
        return ("existing", k, 0)
    mem.append(z)
    return ("new", len(mem) - 1, 0)


def neural_episode(model, batch, cfg, known_cat_index, known_cats, mode="b1", memory=None):
    """B1: first-frame decision, no accumulation, no defer, constant r_phys.
    B2: GRU accumulation, no defer, constant r_phys.
    B3: full core (defer + r_phys conditioning).
    Returns (outcomes, steps_by_age, commit_records, slot_cat).
    Outcomes: (kind_name, payload, commit_age) with payload = predicted known
    cat / slot provenance cat / track cat; None if unresolved."""
    memory = memory or NovelMemory(batch["feats"].device)
    n_occ = batch["feats"].shape[1]
    ep_known_cats = list(batch["pseudo_known"])
    ep_known_idx = [known_cat_index[c] for c in ep_known_cats]
    n_known = len(ep_known_cats)
    outcomes = []
    steps_by_age = {0: 0, 1: 0, 2: 0, 3: 0}
    commit_records = []
    slot_cat = []  # memory slot provenance category, parallel to memory slots
    use_defer = mode == "b3"
    use_r_phys = mode == "b3"
    accumulate = mode in ("b2", "b3")
    for i in range(n_occ):
        role, first = batch["role_first"][i]
        n = int(batch["mask"][i].sum())
        if n == 0:
            outcomes.append(None)
            continue
        h, m = model.belief_init(1, batch["feats"].device)
        first_commit = None
        last_commit = None
        for t in range(n):
            z = model.encode(batch["feats"][i, t : t + 1])
            r = batch["r_phys"][i, t : t + 1].unsqueeze(-1) if use_r_phys else torch.ones(1, 1, device=z.device)
            if accumulate:
                h, m, _ = model.belief_step(z, r, h, m, t)
            else:
                h = z
            age = torch.tensor([[float(t + 1)]], device=h.device)
            logits, lsm = model.decision(h, ep_known_idx, memory, r, age)
            if not use_defer:
                lsm = lsm.clone()
                lsm[0, -1] = -float("inf")
            action = int(lsm[0].argmax())
            kind = model_action_kind(action, n_known, memory.size())
            bucket = 0 if t == 0 else (1 if t == 1 else (2 if t == 2 else 3))
            track_cat = int(batch["cats"][i])
            if kind[0] == "defer":
                steps_by_age[bucket] += 1
            else:
                if kind[0] == "known":
                    payload = ep_known_cats[kind[1]]
                elif kind[0] == "existing":
                    payload = slot_cat[kind[1]]
                else:
                    payload = track_cat
                last_commit = (kind[0], payload, t)
                if first_commit is None:
                    first_commit = (kind[0], payload, t)
                commit_records.append((i, t, kind[0], float(r[0, 0]), track_cat))
                if kind[0] == "new":
                    memory.create(h, float(r[0, 0]), {
                        "role": role, "cat": track_cat, "frame": t,
                    })
                    slot_cat.append(track_cat)
                elif kind[0] == "existing":
                    memory.update(kind[1], h, float(r[0, 0]))
        outcomes.append({"first": first_commit, "last": last_commit})
    return outcomes, steps_by_age, commit_records, slot_cat


def score_outcomes(outcomes, batch, cfg):
    """Score per-occurrence outcomes vs teacher identity labels."""
    cat_to_teacher, n_teacher = {}, 0
    targets, _ = teacher_targets_for_episode(batch, cfg, cat_to_teacher, n_teacher)
    n_occ = batch["feats"].shape[1]
    stats = {
        "known_correct": 0, "known_total": 0, "known_to_novel": 0,
        "novel_first_correct": 0, "novel_first_total": 0,
        "novel_later_correct": 0, "novel_later_total": 0,
        "wrong_reuse": 0, "overbirth": 0, "novel_to_known": 0,
        "fp_commit": 0, "fp_total": 0, "fp_born_slots": 0,
        "unresolved": 0,
        "existing_vs_new_total": 0, "existing_vs_new_correct": 0,
    }
    for i, out in enumerate(outcomes):
        if int(batch["mask"][i].sum()) == 0:
            continue
        if out is None:
            out = {"first": None, "last": None}
        out = out.get("first") if isinstance(out, dict) else out
        role, first = batch["role_first"][i]
        cat = int(batch["cats"][i])
        if role == "known":
            stats["known_total"] += 1
            if out is not None and out[0] == "known" and out[1] == cat:
                stats["known_correct"] += 1
            elif out is not None and out[0] in ("existing", "new"):
                stats["known_to_novel"] += 1
        elif role == "novel":
            if first:
                stats["novel_first_total"] += 1
                if out is not None and out[0] == "new":
                    stats["novel_first_correct"] += 1
                elif out is not None and out[0] == "existing":
                    stats["wrong_reuse"] += 1
                elif out is not None and out[0] == "known":
                    stats["novel_to_known"] += 1
                else:
                    stats["unresolved"] += 1
            else:
                stats["novel_later_total"] += 1
                stats["existing_vs_new_total"] += 1
                if out is None:
                    stats["unresolved"] += 1
                elif out[0] == "existing" and out[1] == cat:
                    stats["novel_later_correct"] += 1
                    stats["existing_vs_new_correct"] += 1
                elif out[0] == "existing":
                    stats["wrong_reuse"] += 1
                elif out[0] == "new":
                    stats["overbirth"] += 1
                elif out[0] == "known":
                    stats["novel_to_known"] += 1
        else:  # fp
            stats["fp_total"] += 1
            if out is not None:
                stats["fp_commit"] += 1
                if out[0] == "new":
                    stats["fp_born_slots"] += 1
    return stats, targets
