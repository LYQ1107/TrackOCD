"""Strictly-online hierarchical episode runtime (train/eval) for Phase 4T."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.iclr27_phase4s.model import NovelMemory


def teacher_targets(batch: dict, cfg):
    """Per-occurrence per-step hierarchical targets.
    Returns (l1_targets, l2_targets, cat_to_teacher, n_teacher).
    l1: 0=KNOWN 1=NOVEL 2=DEFER; l2 (for NOVEL steps): ('new',) /
    ('existing', teacher_slot) / ('defer',)."""
    cat_to_teacher: dict[int, int] = {}
    n_teacher = 0
    l1s, l2s = [], []
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
        seq1, seq2 = [], []
        committed = False
        for t in range(n):
            final = t == n - 1
            age = t + 1
            if role == "fp":
                seq1.append(2)
                seq2.append(("defer",))
                continue
            defer_now = (not final) and age < cfg.min_commit_age
            if role == "known":
                seq1.append(2 if defer_now else 0)
                seq2.append(("defer",))
            else:
                seq1.append(2 if defer_now else 1)
                if first:
                    a = ("new",)
                elif committed:
                    a = ("existing", tid)
                else:
                    a = ("existing", tid) if not first else ("new",)
                seq2.append(("defer",) if defer_now else a)
                if not defer_now:
                    committed = True
        l1s.append(seq1)
        l2s.append(seq2)
    return l1s, l2s, cat_to_teacher, n_teacher


def run_episode(model, batch, cfg, known_cat_index, known_list, memory,
                teacher_state, mode="train"):
    cat_to_teacher = dict(teacher_state["cat_to_teacher"])
    n_teacher = teacher_state["n_teacher"]
    teacher_to_mem = dict(teacher_state["teacher_to_mem"])
    l1s, l2s, cat_to_teacher, n_teacher = teacher_targets(batch, cfg)
    ep_known_cats = list(batch["pseudo_known"])
    ep_known_idx = [known_cat_index[c] for c in ep_known_cats]
    ep_known_pos = {c: i for i, c in enumerate(ep_known_cats)}
    n_known = len(ep_known_cats)
    losses = {"l1": [], "known": [], "l2": [], "mem_pull": [], "mem_push": [], "contrast": []}
    h_finals = []
    decisions = []
    q_cpu = batch.get("qphys")
    r_cpu = batch.get("r_phys")
    if q_cpu is not None:
        q_cpu = q_cpu.cpu().numpy()
    n_per_occ = batch["mask"].sum(-1).cpu().numpy().tolist()
    cats_cpu = batch["cats"].cpu().numpy().tolist()

    for i in range(batch["feats"].shape[1]):
        n = int(n_per_occ[i])
        h, m = model.belief_init(1, batch["feats"].device)
        h_final = None
        for t in range(n):
            z = model.encode(batch["feats"][i, t : t + 1])
            use_qphys = bool(getattr(model, "use_qphys", False)) and q_cpu is not None
            if use_qphys:
                r = torch.tensor([q_cpu[i, t].tolist()], device=h.device)
            else:
                r = torch.tensor([[float(r_cpu[i, t])]], device=h.device)
            h, m, g = model.belief_step(z, r, h, m, t)
            age = torch.tensor([[float(t + 1)]], device=h.device)
            out = model.decision(h, ep_known_idx, memory, r, age)
            l1_t = l1s[i][t]
            l2_t = l2s[i][t]
            if mode == "train":
                # In forced-decision mode the DEFER logit is clamped to -inf,
                # so DEFER targets carry no learnable signal; skip them.
                if not (l1_t == 2 and not getattr(model, "use_defer", True)):
                    losses["l1"].append(out["l1_lsm"][0, l1_t].unsqueeze(0))
                if l1_t == 0:  # KNOWN
                    kl = out["known"]
                    losses["known"].append(-F.log_softmax(kl, dim=-1)[0, ep_known_pos[int(cats_cpu[i])]].unsqueeze(0))
                elif l1_t == 1:  # NOVEL
                    kind = l2_t[0]
                    nl = out["l2"]["novel"]
                    if kind == "new":
                        # Level-2 fixed 3-way gate: 0=existing,1=new,2=defer
                        losses["l2"].append(out["l2_lsm"][0, 1].unsqueeze(0))
                    elif kind == "existing":
                        k = teacher_to_mem[l2_t[1]]
                        losses["l2"].append(out["l2_lsm"][0, 0].unsqueeze(0))
                        if nl.shape[1] >= 1:
                            losses["l2"].append(
                                F.log_softmax(nl, dim=-1)[0, k].unsqueeze(0))
                    else:
                        losses["l2"].append(out["l2_lsm"][0, 2].unsqueeze(0))
                # teacher-forced memory writes
                if l1_t == 1 and l2_t[0] == "new":
                    tid = cat_to_teacher[int(cats_cpu[i])]
                    k = memory.create(h, float(r_cpu[i, t]), {"cat": int(cats_cpu[i])})
                    teacher_to_mem[tid] = k
                    hn = F.normalize(h, dim=-1)
                    if memory.size() > 1:
                        cos = hn @ memory.protos.detach().clone()[:-1].t()
                        losses["mem_push"].append(torch.relu(cos - 0.45).mean().reshape(1))
                elif l1_t == 1 and l2_t[0] == "existing":
                    k = teacher_to_mem[l2_t[1]]
                    memory.update(k, h, float(r_cpu[i, t]))
                    hn = F.normalize(h, dim=-1)
                    pull = 1.0 - F.cosine_similarity(hn, memory.protos.detach().clone()[k:k + 1]).reshape(1)
                    losses["mem_pull"].append(pull)
                    if memory.size() > 1:
                        others = [j for j in range(memory.size()) if j != k]
                        cos = hn @ memory.protos.detach().clone()[others].t()
                        losses["mem_push"].append(torch.relu(cos - 0.35).mean().reshape(1))
                decisions.append((i, t, l1_t, l2_t))
            else:
                a1 = int(out["l1_lsm"][0].argmax())
                a2 = None
                if a1 == 1:
                    a2 = int(out["l2_lsm"][0].argmax())
                    if a2 == 0:
                        slot = int(out["l2"]["novel"][0].argmax())
                        memory.update(slot, h, float(r_cpu[i, t]))
                    elif a2 == 1:
                        memory.create(h, float(r_cpu[i, t]), {"cat": int(cats_cpu[i])})
                decisions.append((i, t, a1, a2))
            h_final = h
        h_finals.append(h_final)

    novel_idx = [i for i in range(batch["feats"].shape[1])
                 if batch["role_first"][i][0] == "novel"]
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
    out = {"decisions": decisions, "h_finals": h_finals}
    if mode == "train":
        for k in losses:
            out[k] = torch.cat(losses[k]) if losses[k] else None
    return out
