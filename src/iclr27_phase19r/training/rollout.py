"""Causal mixed-episode rollout and risk-aware losses."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.iclr27_phase19r.data.episodes import MetaEpisode, StreamItem
from src.iclr27_phase19r.runtime.state import StateMemory
from src.iclr27_phase19r.runtime.runner import risk_decode


def teacher_probability(step: int, total: int) -> float:
    if step <= 4000:
        return 1.0 - .15 * (step - 1) / 3999.0
    if step <= 16000:
        return .85 - .65 * (step - 4000) / 12000.0
    return 0.0


def _bundle_batch(memories: list[StateMemory], raws: list[np.ndarray] | torch.Tensor, items: list[StreamItem], device: torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if isinstance(raws, torch.Tensor):
        raw_rows = [raws[i] for i in range(raws.shape[0])]
    else:
        raw_rows = [torch.from_numpy(r) for r in raws]
    bundles = [m.build_candidate_tensors(r, it.video_id, it.track_key, device=device) for m, r, it in zip(memories, raw_rows, items)]
    maxc = max((len(x["state_indices"]) for x in bundles), default=0)
    b = len(bundles); d = raws[0].shape[-1]
    sr = torch.zeros(b, maxc, d, device=device); sz = torch.zeros_like(sr); sf = torch.zeros(b, maxc, 6, device=device); sm = torch.zeros(b, maxc, dtype=torch.bool, device=device)
    for i, x in enumerate(bundles):
        n = len(x["state_indices"])
        if n:
            sr[i, :n] = x["state_raw"][0]; sz[i, :n] = x["state_z"][0]; sf[i, :n] = x["state_features"][0]; sm[i, :n] = True
    return {"state_raw": sr, "state_z": sz, "state_features": sf, "state_mask": sm}, bundles


def _target(item: StreamItem, ep: MetaEpisode, memory: StateMemory, bundle: dict[str, Any], data: Any, max_states: int) -> tuple[int, str, int | None, list[int]]:
    k = len(data.supported_ids); cat = item.oracle_category_for_loss_only
    if item.role == "visible_known" and cat in data.known_to_index:
        j = data.known_to_index[int(cat)]
        if bool(ep.known_mask[j]):
            return j, "KNOWN", None, []
    if item.role == "legal_unlabeled" or item.target_kind == "DEFER":
        return k + max_states + 1, "DEFER", None, []
    matches = [j for j, s in enumerate(memory.states) if cat is not None and s.oracle_birth_category == int(cat)]
    candidate_pos = [bundle["state_indices"].index(j) for j in matches if j in bundle["state_indices"]]
    if candidate_pos:
        return k + candidate_pos[0], "EXISTING", matches[0], candidate_pos
    return k + max_states, "NEW", None, []


def rollout_batch(model: Any, data: Any, episodes: list[MetaEpisode], device: torch.device,
                  step: int, total_steps: int, rng: np.random.Generator,
                  *, train: bool = True, ladder: str = "L2", allow_defer: bool = True,
                  teacher_probability_override: float | None = None,
                  event_commit_weight: float = 0.0) -> tuple[torch.Tensor, dict[str, float], dict[str, int], list[list[dict[str, Any]]]]:
    # Training keeps causal state transitions identical but avoids serializing
    # full snapshots/anchor tensors and keeps dispersion updates on-device.
    # Evaluator/diagnostic rollouts retain the fully materialized trace.
    memories = [StateMemory(max_states=model.max_states, max_anchors=8,
                            fast_mode=bool(train), record_trace=not bool(train)) for _ in episodes]
    p_teacher = float(teacher_probability_override) if teacher_probability_override is not None else (teacher_probability(step, total_steps) if train else 0.0)
    losses: dict[str, list[torch.Tensor]] = defaultdict(list); counts: Counter[str] = Counter(); traces: list[list[dict[str, Any]]] = [[] for _ in episodes]
    mask_active_sum = 0; mask_total = 0
    for t in range(len(episodes[0].items)):
        items = [ep.items[t] for ep in episodes]; raws = [x.raw for x in items]
        raw_t = torch.from_numpy(np.stack(raws)).to(device); geom_t = torch.from_numpy(np.stack([x.geom for x in items])).to(device); q_t = torch.tensor([x.quality for x in items], dtype=torch.float32, device=device)
        bundle, bmeta = _bundle_batch(memories, raw_t, items, device)
        known_mask = torch.from_numpy(np.stack([ep.known_mask for ep in episodes])).to(device)
        mask_active_sum += int(known_mask.sum().item()); mask_total += int(known_mask.numel())
        out = model(raw_t, geom_t, q_t, known_mask, bundle, allow_defer=allow_defer)
        targets = []; target_kinds = []; target_global = []; candidate_pos_all = []
        for i, (ep, item, bm) in enumerate(zip(episodes, items, bmeta)):
            ti, kind, global_idx, cpos = _target(item, ep, memories[i], bm, data, model.max_states)
            targets.append(ti); target_kinds.append(kind); target_global.append(global_idx); candidate_pos_all.append(cpos)
        target_t = torch.tensor(targets, dtype=torch.long, device=device)
        ce = F.cross_entropy(out["logits"], target_t, reduction="none")
        # False merge errors cost more than ordinary misses.
        action_w = torch.tensor([3.0 if k == "NEW" and len(memories[i].states) > 0 else 1.0 for i, k in enumerate(target_kinds)], device=device)
        losses["action_ce"].append((ce * action_w).mean())
        # Corrective event-aligned training can explicitly move post-reliable
        # commits above the DEFER logit.  This targets the audited dominant
        # failure (unresolved/over-defer) without changing the evaluator or
        # adding a memory component.  Mixed episodes and the registered run
        # retain the original loss when this weight is zero.
        if event_commit_weight > 0.0:
            defer_idx = len(data.supported_ids) + model.max_states + 1
            margins = []
            for i, (ep, item, kind) in enumerate(zip(episodes, items, target_kinds)):
                if "event_aligned" not in str(ep.episode_id) or kind not in {"NEW", "EXISTING"}:
                    continue
                commit_logits = out["logits"][i, :len(data.supported_ids) + model.max_states + 1]
                margins.append(F.softplus(out["logits"][i, defer_idx] - commit_logits.max()))
            if margins:
                losses["event_commit_margin"].append(torch.stack(margins).mean())
            else:
                losses["event_commit_margin"].append(out["logits"].sum() * 0.0)
        # Balanced candidate same/different loss, with hard-negative emphasis.
        cand_vals = []; cand_lab = []; cand_w = []
        for i, (item, bm) in enumerate(zip(items, bmeta)):
            n = len(bm["state_indices"]); cat = item.oracle_category_for_loss_only
            for j in range(n):
                st = memories[i].states[bm["state_indices"][j]]
                label = float(cat is not None and st.oracle_birth_category == int(cat))
                cand_vals.append(out["candidate_score"][i, j]); cand_lab.append(label); cand_w.append(2.5 if item.hard_negative and label == 0 else 1.0)
        if cand_vals:
            cv = torch.stack(cand_vals); cy = torch.tensor(cand_lab, device=device); cw = torch.tensor(cand_w, device=device)
            losses["candidate_same_different"].append((F.binary_cross_entropy_with_logits(cv, cy, reduction="none") * cw).sum() / cw.sum().clamp_min(1.))
            losses["false_merge_risk"].append((F.softplus(cv) * (1 - cy) * cw).sum() / cw.sum().clamp_min(1.))
        else:
            losses["candidate_same_different"].append(out["logits"].sum() * 0.0); losses["false_merge_risk"].append(out["logits"].sum() * 0.0)
        # NEW-vs-EXISTING preference from non-empty memory.
        ne_vals = []; ne_targets = []
        for i, kind in enumerate(target_kinds):
            if kind in {"NEW", "EXISTING"} and out["candidate_score"].shape[1]:
                best = out["candidate_score"][i].max(); ne_vals.append(best - out["new_logit"][i]); ne_targets.append(float(kind == "EXISTING"))
        if ne_vals:
            losses["new_existing"].append(F.binary_cross_entropy_with_logits(torch.stack(ne_vals), torch.tensor(ne_targets, device=device)))
        else:
            losses["new_existing"].append(out["logits"].sum() * 0.0)
        visible = torch.tensor([k == "KNOWN" for k in target_kinds], dtype=torch.bool, device=device)
        if visible.any():
            losses["known_calibration"].append(F.cross_entropy(out["known_logits"][visible], target_t[visible]))
        else:
            losses["known_calibration"].append(out["logits"].sum() * 0.0)
        q_target = torch.tensor([x.quality for x in items], dtype=torch.float32, device=device)
        losses["quality"].append(F.mse_loss(out["quality"], q_target))
        # Model action or scheduled teacher action drives the next state.
        for i, (ep, item, bm) in enumerate(zip(episodes, items, bmeta)):
            use_teacher = bool(train and rng.random() < p_teacher)
            if use_teacher:
                kind = target_kinds[i]; global_idx = target_global[i];
                if kind == "KNOWN": action, state_idx = "KNOWN", None
                elif kind == "EXISTING": action, state_idx = "EXISTING", global_idx
                elif kind == "DEFER": action, state_idx = "DEFER", None
                else: action, state_idx = "NEW", None
                conf = 1.0
            else:
                action, local_idx, conf, _ = risk_decode(out, bm, item.quality, known_mask[i], model.known_count, model.max_states, allow_defer, model.tau_ready, model.tau_known, model.tau_assign)
                state_idx = bm["state_indices"][local_idx] if action == "EXISTING" and local_idx is not None and local_idx < len(bm["state_indices"]) else None
            rec = memories[i].apply_action(action, raw_t[i], out["z"][i], item.video_id, item.track_key,
                                           state_index=state_idx, oracle_category=item.oracle_category_for_loss_only,
                                           quality=item.quality, confidence=float(conf), update_allowed=True)
            rec["target_kind"] = target_kinds[i]; rec["teacher"] = use_teacher; rec["hard_negative"] = item.hard_negative
            traces[i].append(rec); counts["teacher" if use_teacher else "model"] += 1; counts[action] += 1
    weights = {"action_ce": 1.0, "candidate_same_different": 1.0, "new_existing": .75,
               "known_calibration": .35, "quality": .20, "false_merge_risk": 1.5,
               "event_commit_margin": float(event_commit_weight)}
    total = sum(weights[k] * torch.stack(v).mean() for k, v in losses.items() if k in weights)
    scalars = {k: float(torch.stack(v).mean().detach()) for k, v in losses.items()}; scalars.update({"total": float(total.detach()), "teacher_probability": float(p_teacher), "on_policy_fraction": float(1.0 - p_teacher), "known_mask_active_mean": float(mask_active_sum / max(len(episodes) * len(episodes[0].items), 1)), "known_mask_masked_mean": float((mask_total - mask_active_sum) / max(len(episodes) * len(episodes[0].items), 1)), "known_mask_active_fraction": float(mask_active_sum / max(mask_total, 1))})
    return total, scalars, dict(counts), traces
