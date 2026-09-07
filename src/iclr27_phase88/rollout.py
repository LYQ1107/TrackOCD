"""Persistent TRAIN rollout; it shares StateMemoryV2/transitions with runtime."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase88.controller import CausalPersistentOCD, decode_joint_action, mask_joint_logits
from src.iclr27_phase88.data import FeatureStore
from src.iclr27_phase88.memory import StateMemoryV2, TargetSession
from src.iclr27_phase88.support_evidence import compute_support_evidence
from src.iclr27_phase88.transitions import apply_session_action, finalize_session


def _target_action(session: TargetSession, memory: StateMemoryV2, event: dict[str, Any],
                   role: str, category: int, position: int, reliable: int,
                   candidate_indices: list[int]) -> tuple[str, int | None, int | None, int | None]:
    positive = event.get("polarity") == "positive"
    if session.committed_action is not None:
        correct_existing = (
            positive and session.committed_action == "EXISTING" and
            session.committed_global_index is not None and
            session.committed_global_index < len(memory.states) and
            memory.states[session.committed_global_index].oracle_birth_category == category
        )
        correct_new = (not positive and session.committed_action == "NEW")
        if not (correct_existing or correct_new):
            return "RESET", None, None, None
        return session.committed_action, session.committed_state_index, session.committed_known_index, session.committed_global_index
    if role == "source":
        matches = [i for i in candidate_indices if memory.states[i].oracle_birth_category == category]
        if matches:
            return "EXISTING", candidate_indices.index(matches[0]), None, matches[0]
        return "NEW", None, None, None
    if position + 1 < reliable:
        return "DEFER", None, None, None
    if positive:
        matches = [i for i in candidate_indices if memory.states[i].oracle_birth_category == category]
        if matches:
            return "EXISTING", candidate_indices.index(matches[0]), None, matches[0]
        # This should only occur when a source teacher action was unavailable;
        # NEW remains legal and exposes the missing-support failure.
        return "NEW", None, None, None
    return "NEW", None, None, None


def _target_index(model: CausalPersistentOCD, action: str, slot: int | None, known: int | None) -> int:
    if action == "KNOWN":
        if known is None:
            raise RuntimeError("KNOWN target requires known index")
        return int(known)
    if action == "EXISTING":
        if slot is None:
            raise RuntimeError("EXISTING target requires state slot")
        return model.known_count + int(slot)
    if action == "NEW":
        return model.known_count + model.max_states
    if action == "DEFER":
        return model.known_count + model.max_states + 1
    if action == "RESET":
        return model.known_count + model.max_states + 2
    raise ValueError(action)


def _rollout_track(model: CausalPersistentOCD, store: FeatureStore, memory: StateMemoryV2,
                   event: dict[str, Any], track_key: str, role: str, category: int,
                   known_mask: torch.Tensor, teacher_probability: float, rng: random.Random,
                   train: bool, losses: dict[str, list[torch.Tensor]], trace: list[dict[str, Any]],
                   support_mode: bool) -> TargetSession:
    device = next(model.parameters()).device
    array = store.track(track_key)
    n = min(16, len(array.raw))
    raw = torch.from_numpy(array.raw[:n][None]).to(device)
    geom = torch.from_numpy(array.geom[:n][None]).to(device)
    quality = torch.from_numpy(array.quality[:n][None]).to(device)
    encoded = model.encode_track(raw, geom, torch.ones((1, n), dtype=torch.bool, device=device))
    session = TargetSession(track_key, store.video(track_key))
    if role == "target" and train and event.get("reset_injection"):
        wrong = memory.candidate_indices(store.video(track_key), track_key)
        if wrong:
            session.committed_action = "EXISTING"
            session.committed_state_index = 0
            session.committed_global_index = wrong[0]
    reliable = max(1, min(16, int(event.get("reliable_prefix_for_loss_only", n))))
    for position in range(n):
        tensors = memory.build_candidate_tensors(store.video(track_key), track_key)
        previous = session.evidence if session.evidence is not None else torch.zeros((1, model.max_states), device=device)
        q = quality[:, position]
        support = compute_support_evidence(
            encoded["semantic_seq"][:, position], tensors["prototypes"], tensors["prototype_mask"],
            tensors["state_stats"], q, torch.tensor([session.best_streak], device=device),
        ) if support_mode else torch.zeros((1, 8), device=device)
        output = model.forward_action(
            encoded["semantic_seq"][:, position], encoded["track_hidden_seq"][:, position],
            tensors["prototypes"], tensors["prototype_mask"], tensors["state_stats"],
            previous, support, q, torch.tensor([session.best_streak], device=device),
            known_mask,
        )
        logits = mask_joint_logits(
            output["joint_logits"], model.known_count, model.max_states,
            [session.committed_action], [session.committed_state_index], [session.committed_known_index],
        )
        desired, desired_slot, desired_known, desired_global = _target_action(
            session, memory, event, role, category, position, reliable, tensors["state_indices"]
        )
        target_index = _target_index(model, desired, desired_slot, desired_known)
        if float(logits[0, target_index].detach().cpu()) <= -1e3:
            raise RuntimeError("TARGET_ACTION_MASKED_BY_STATE_MACHINE")
        losses["action_ce"].append(F.cross_entropy(logits, torch.tensor([target_index], device=device)))
        valid = tensors["prototype_mask"].any(dim=-1)
        relation_target = torch.zeros_like(output["state_logits"])
        if desired == "EXISTING" and desired_slot is not None and desired_slot < relation_target.shape[1]:
            relation_target[:, desired_slot] = 1.0
        losses["state_relation"].append(F.binary_cross_entropy_with_logits(output["state_logits"], relation_target))
        if desired != "EXISTING":
            losses["false_merge_risk"].append(2.0 * torch.sigmoid(output["state_logits"].masked_fill(~valid, -1e4)).mean())
        if desired == "EXISTING":
            losses["new_existing_margin"].append(0.75 * F.softplus(output["new_logit"] - output["state_logits"][:, int(desired_slot)] + 0.2).mean())
        else:
            best_state = output["state_logits"].masked_fill(~valid, -1e4).max(dim=-1).values
            losses["new_existing_margin"].append(0.75 * F.softplus(best_state - output["new_logit"] + 0.2).mean())
        if desired == "DEFER":
            losses["commit_defer_margin"].append(0.75 * F.softplus(output["new_logit"] - output["defer_logit"] + 0.2).mean())
        else:
            correct_logit = output["state_logits"][:, int(desired_slot)] if desired == "EXISTING" and desired_slot is not None else output["new_logit"]
            losses["commit_defer_margin"].append(0.75 * F.softplus(output["defer_logit"] - correct_logit + 0.2).mean())
        if desired == "RESET":
            non_reset = torch.cat([output["new_logit"].reshape(1, 1), output["defer_logit"].reshape(1, 1), output["state_logits"].masked_fill(~valid, -1e4)], dim=1).max(dim=-1).values
            losses["reset_margin"].append(0.75 * F.softplus(non_reset - output["reset_logit"] + 0.2).mean())
        with torch.no_grad():
            predicted, pred_slot, confidence = decode_joint_action(logits, model.known_count, model.max_states)
        chosen = desired if train and rng.random() < teacher_probability else predicted
        chosen_slot = desired_slot if chosen == "EXISTING" and train and rng.random() < teacher_probability else pred_slot
        chosen_known = desired_known if chosen == "KNOWN" and train else (pred_slot if predicted == "KNOWN" else None)
        chosen_global = desired_global if chosen == "EXISTING" and chosen_slot == desired_slot else (
            tensors["state_indices"][chosen_slot] if chosen == "EXISTING" and chosen_slot is not None and chosen_slot < len(tensors["state_indices"]) else None
        )
        apply_session_action(session, chosen, chosen_slot if chosen == "EXISTING" else None,
                             chosen_known, chosen_global)
        if session.evidence is None:
            session.evidence = torch.zeros_like(previous)
        session.evidence = 0.70 * session.evidence + 0.30 * torch.sigmoid(output["state_logits"])
        valid_now = tensors["prototype_mask"].any(dim=-1).squeeze(0)
        best = int(torch.argmax(output["state_logits"], dim=-1).item()) if bool(valid_now.any()) else None
        if best is None:
            session.best_streak = 0; session.previous_best_index = None
        elif best == session.previous_best_index:
            session.best_streak += 1
        else:
            session.best_streak = 1; session.previous_best_index = best
        trace.append({"track_key": track_key, "role": role, "position": position + 1, "target": desired, "predicted": predicted, "chosen": chosen, "confidence": confidence, "target_index": target_index, "state_slot": desired_slot, "reset": int(chosen == "RESET"), "best_streak": int(session.best_streak)})
    if session.collected_vectors:
        final_vector = torch.mean(encoded["semantic_seq"][0, :n], dim=0)
    else:
        final_vector = encoded["semantic"] .squeeze(0)
    session.collected_vectors.extend(encoded["semantic_seq"][0, :n].detach().cpu().numpy())
    finalize_session(memory, session, final_vector, n, category)
    return session


def rollout_known_event(model: CausalPersistentOCD, store: FeatureStore, key: str,
                        category: int, known_mask: torch.Tensor,
                        known_index: int | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    device = next(model.parameters()).device
    arr = store.track(key); n = min(16, len(arr.raw))
    raw = torch.from_numpy(arr.raw[:n][None]).to(device); geom = torch.from_numpy(arr.geom[:n][None]).to(device)
    quality = torch.from_numpy(arr.quality[:n][None]).to(device)
    enc = model.encode_track(raw, geom, torch.ones((1, n), dtype=torch.bool, device=device))
    prev = torch.zeros((1, model.max_states), device=device)
    loss_terms = []
    target = int(known_index) if known_index is not None else None
    trace = []
    for p in range(n):
        prototypes = torch.zeros((1, model.max_states, 4, 768), device=device)
        pmask = torch.zeros((1, model.max_states, 4), dtype=torch.bool, device=device)
        stats = torch.zeros((1, model.max_states, 6), device=device)
        support = torch.zeros((1, 8), device=device)
        out = model.forward_action(enc["semantic_seq"][:, p], enc["track_hidden_seq"][:, p], prototypes, pmask, stats, prev, support, quality[:, p], torch.zeros((1,), device=device), known_mask)
        slot = target
        if slot is None and known_mask.bool().any():
            slot = int(torch.where(known_mask.bool().reshape(-1))[0][0])
        if slot is not None:
            loss_terms.append(F.cross_entropy(out["joint_logits"], torch.tensor([slot], device=device)))
            others = out["joint_logits"].clone(); others[:, slot] = -1e4
            loss_terms.append(0.75 * F.softplus(others.max(dim=-1).values - out["joint_logits"][:, slot] + 0.2).mean())
            trace.append({"position": p + 1, "action": "KNOWN", "known_index": slot})
    return (sum(loss_terms) / max(len(loss_terms), 1), {"trace": trace, "known_category": int(category)})


def rollout_event(model: CausalPersistentOCD, store: FeatureStore, event: dict[str, Any],
                  train: bool, teacher_probability: float = 0.0,
                  rng: random.Random | None = None, support_mode: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
    device = next(model.parameters()).device
    data = store.data
    mask_np = np.asarray(data.active_known_mask, dtype=bool).copy()
    for cat in event.get("masked_known_categories_for_loss_only", []):
        j = data.known_to_index.get(int(cat))
        if j is not None:
            mask_np[j] = False
    known_mask = torch.from_numpy(mask_np).to(device)
    losses: dict[str, list[torch.Tensor]] = {"action_ce": [], "state_relation": [], "false_merge_risk": [], "new_existing_margin": [], "commit_defer_margin": [], "reset_margin": [], "known_margin": []}
    memory = StateMemoryV2(max_states=model.max_states, max_prototypes=4, device=device)
    trace: list[dict[str, Any]] = []
    rng = rng or random.Random(0)
    for source in event["source_tracks"]:
        _rollout_track(model, store, memory, event, source["track_key"], "source", int(source["category_for_loss_only"]), known_mask, teacher_probability, rng, train, losses, trace, support_mode)
    target = event["target_track_key"]
    _rollout_track(model, store, memory, event, target, "target", int(event["target_category_for_loss_only"]), known_mask, teacher_probability, rng, train, losses, trace, support_mode)
    weights = {"action_ce": 1.0, "state_relation": 1.0, "false_merge_risk": 2.0, "new_existing_margin": 0.75, "commit_defer_margin": 0.75, "reset_margin": 0.75, "known_margin": 0.75}
    present = {k: sum(v) / max(len(v), 1) for k, v in losses.items() if v}
    total = sum(weights[k] * v for k, v in present.items())
    return total, {"event_id": event.get("event_id"), "polarity": event.get("polarity"), "trace": trace, "losses": {k: float(v.detach().cpu()) for k, v in present.items()}, "state_count": len(memory.states), "reset_targets": sum(int(x["target"] == "RESET") for x in trace), "reset_predictions": sum(int(x["chosen"] == "RESET") for x in trace)}
