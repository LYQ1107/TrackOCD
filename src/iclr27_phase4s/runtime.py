"""Shared strictly-online episode runtime for training and evaluation.

Frame t only sees <= t evidence. Teacher forcing is used for memory WRITES
during training (labels reference teacher-consistent slots); at eval the
memory evolves from the model's own actions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.iclr27_phase4s.model import NovelMemory, SemanticCore


def teacher_targets_for_episode(batch: dict, cfg, cat_to_teacher: dict[int, int], n_teacher: int):
    """Return per-occurrence per-step targets as (kind, payload):
    ('known', cat) / ('existing', teacher_slot_id) / ('new',) / ('defer',).
    Also returns updated cat_to_teacher + next teacher id."""
    targets = []
    for i in range(batch["feats"].shape[1]):
        role, first = batch["role_first"][i]
        n = int(batch["mask"][i].sum())
        cat = int(batch["cats"][i])
        if role == "novel" and first:
            tid = n_teacher
            n_teacher += 1
            cat_to_teacher[cat] = tid
        elif role == "novel":
            tid = cat_to_teacher.get(cat, -1)
        else:
            tid = -1
        seq = []
        committed = False
        for t in range(n):
            final = t == n - 1
            age = t + 1
            r_phys = float(batch["r_phys"][i, t])
            if role == "fp":
                seq.append(("defer",))
            else:
                if role == "known":
                    a = ("known", cat)
                elif committed:
                    a = ("existing", tid)  # own slot after first commit
                elif first:
                    a = ("new",)
                else:
                    a = ("existing", tid)
                if not final and (age < cfg.min_commit_age or r_phys < cfg.r_phys_floor):
                    seq.append(("defer",))
                else:
                    seq.append(a)
                    committed = True
        targets.append(seq)
    return targets, n_teacher


def action_index(action, known_pos: dict[int, int], n_known: int, teacher_to_mem: dict[int, int], n_slots: int) -> int:
    """Map teacher target to the index in the concatenated action space."""
    kind = action[0]
    if kind == "known":
        return known_pos[action[1]]
    if kind == "new":
        return n_known + n_slots
    if kind == "defer":
        return n_known + n_slots + 1
    # existing(teacher_slot)
    return n_known + teacher_to_mem[action[1]]


def run_episode(
    model: SemanticCore,
    batch: dict,
    cfg,
    known_cat_index: dict[int, int],
    known_cats: list[int],
    memory: NovelMemory,
    teacher_state: dict,
    mode: str = "train",
):
    """Process one episode occurrence-by-occurrence, strictly online.

    mode == 'train': memory writes teacher-forced by targets, grads enabled.
    mode == 'eval': model-driven memory, no grads.

    Returns dict with losses (train), decisions, targets, h_T, stats.
    """
    cat_to_teacher = dict(teacher_state["cat_to_teacher"])
    n_teacher = teacher_state["n_teacher"]
    teacher_to_mem = dict(teacher_state["teacher_to_mem"])
    targets, n_teacher = teacher_targets_for_episode(batch, cfg, cat_to_teacher, n_teacher)

    n_occ = batch["feats"].shape[1]
    ep_known_cats = list(batch["pseudo_known"])
    ep_known_idx = [known_cat_index[c] for c in ep_known_cats]
    ep_known_pos = {c: i for i, c in enumerate(ep_known_cats)}
    n_known = len(ep_known_cats)
    all_logits = []
    decisions = []
    h_finals = []
    losses = {"decision": [], "known": [], "mem_pull": [], "mem_push": [], "contrast": []}
    commit_records = []  # (occ_idx, step, action, target, h, r_phys)
    # CPU copies avoid one device sync per step (big speedup for the B=1 loop)
    r_cpu = batch["r_phys"].cpu().numpy().tolist()
    n_per_occ = batch["mask"].sum(-1).cpu().numpy().tolist()
    cats_cpu = batch["cats"].cpu().numpy().tolist()

    for i in range(n_occ):
        n = int(n_per_occ[i])
        h, m = model.belief_init(1, batch["feats"].device)
        h_final = None
        for t in range(n):
            z = model.encode(batch["feats"][i, t : t + 1])
            r_t = float(r_cpu[i][t])
            r = torch.tensor([[r_t]], device=h.device)
            h, m, g = model.belief_step(z, r, h, m, t)
            age = torch.tensor([[float(t + 1)]], device=h.device)
            logits, lsm = model.decision(h, ep_known_idx, memory, r, age)
            all_logits.append(lsm[0])
            action = int(lsm[0].argmax()) if mode == "eval" else -1
            target = targets[i][t]
            decisions.append((i, t, action, target))
            if mode == "train":
                idx = action_index(target, ep_known_pos, n_known, teacher_to_mem, memory.size())
                losses["decision"].append(lsm[0, idx].unsqueeze(0))
                # teacher-forced memory write when the target commits
                kind = target[0]
                if kind == "new":
                    tid = cat_to_teacher[int(cats_cpu[i])]
                    k = memory.create(h, r_t, {
                        "role": batch["role_first"][i][0],
                        "cat": int(cats_cpu[i]),
                        "frame": t,
                    })
                    teacher_to_mem[tid] = k
                    # push away from all pre-existing slots
                    if memory.size() > 1:
                        hn = F.normalize(h, dim=-1)
                        cos = hn @ memory.protos.detach().clone()[:-1].t()
                        losses["mem_push"].append(torch.relu(cos - 0.45).mean().reshape(1))
                elif kind == "existing":
                    k = teacher_to_mem[target[1]]
                    memory.update(k, h, r_t)
                    hn = F.normalize(h, dim=-1)
                    pull = 1.0 - F.cosine_similarity(hn, memory.protos.detach().clone()[k:k + 1]).reshape(1)
                    losses["mem_pull"].append(pull)
                    if memory.size() > 1:
                        others = [j for j in range(memory.size()) if j != k]
                        cos = hn @ memory.protos.detach().clone()[others].t()
                        losses["mem_push"].append(torch.relu(cos - 0.35).mean().reshape(1))
                elif kind == "known":
                    kl = model.known_logits(h, ep_known_idx)
                    idx = ep_known_pos[target[1]]
                    losses["known"].append(-F.log_softmax(kl, dim=-1)[0, idx].unsqueeze(0))
                commit_records.append((i, t, target, action, h.detach(), r_t))
            else:
                kind = model_action_kind(action, n_known, memory.size())
                if kind[0] == "new":
                    memory.create(h, r_t, {
                        "role": batch["role_first"][i][0],
                        "cat": int(cats_cpu[i]),
                        "frame": t,
                    })
                elif kind[0] == "existing":
                    memory.update(kind[1], h, r_t)
                commit_records.append((i, t, target, action, h.detach(), r_t))
            h_final = h
        h_finals.append(h_final)

    # cross-physical-track contrast over pseudo-novel occurrences
    novel_idx = [i for i in range(n_occ) if batch["role_first"][i][0] == "novel"]
    contrast_logits = []
    if len(novel_idx) >= 2 and mode == "train":
        hs = torch.cat([h_finals[i] for i in novel_idx], dim=0)
        hs = F.normalize(hs, dim=-1)
        cats = torch.tensor([int(batch["cats"][i]) for i in novel_idx], device=hs.device)
        sim = hs @ hs.t() / 0.1
        for a in range(len(novel_idx)):
            pos = (cats == cats[a]) & (torch.arange(len(novel_idx), device=hs.device) != a)
            if pos.any():
                num = sim[a, pos].logsumexp(dim=0)
                den = sim[a].logsumexp(dim=0)
                losses["contrast"].append((den - num).unsqueeze(0))

    out = {
        "decisions": decisions,
        "targets": targets,
        "h_finals": h_finals,
        "commit_records": commit_records,
    }
    if mode == "train":
        for k in losses:
            out[k] = torch.cat(losses[k]) if losses[k] else None
    return out


def model_action_kind(action: int, n_known: int, n_slots: int):
    if action < n_known:
        return ("known", action)
    if action < n_known + n_slots:
        return ("existing", action - n_known)
    if action == n_known + n_slots:
        return ("new",)
    return ("defer",)
