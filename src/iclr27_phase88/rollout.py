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
    # Role is resolved before event polarity.  Source tracks are never judged
    # by the positive/negative label of the eventual target event.
    matches = [
        global_index for global_index in candidate_indices
        if 0 <= global_index < len(memory.states)
        and memory.states[global_index].oracle_birth_category == int(category)
    ]

    def local_slot(global_index: int) -> int:
        return candidate_indices.index(global_index)

    if role == "source":
        if session.committed_action is None:
            if matches:
                g = matches[0]
                return "EXISTING", local_slot(g), None, g
            return "NEW", None, None, None
        if session.committed_action == "EXISTING":
            if session.committed_global_index in matches:
                return "EXISTING", session.committed_state_index, None, session.committed_global_index
            return "RESET", None, None, None
        if session.committed_action == "NEW":
            if not matches:
                return "NEW", None, None, None
            return "RESET", None, None, None
        # A pseudo-novel source must not become KNOWN.
        return "RESET", None, None, None

    positive = event.get("polarity") == "positive"
    if session.committed_action is not None:
        if positive:
            correct = session.committed_action == "EXISTING" and session.committed_global_index in matches
            if correct:
                return "EXISTING", session.committed_state_index, None, session.committed_global_index
        else:
            if session.committed_action == "NEW":
                return "NEW", None, None, None
        return "RESET", None, None, None

    if position + 1 < reliable:
        return "DEFER", None, None, None
    if positive:
        if matches:
            g = matches[0]
            return "EXISTING", local_slot(g), None, g
        # A target cannot create a duplicate birth when source materialization
        # failed.  It must defer and expose missing support instead.
        session.missing_positive_state = True
        return "DEFER", None, None, None
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
        candidate_indices = memory.candidate_indices(store.video(track_key), track_key)
        wrong = [
            global_index for global_index in candidate_indices
            if memory.states[global_index].oracle_birth_category is not None
            and int(memory.states[global_index].oracle_birth_category) != int(category)
        ]
        if wrong:
            session.committed_action = "EXISTING"
            session.committed_state_index = candidate_indices.index(wrong[0])
            session.committed_global_index = wrong[0]
            session.reset_injection_applied = True
            session.reset_injection_global_index = int(wrong[0])
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
        elif desired == "NEW":
            best_state = output["state_logits"].masked_fill(~valid, -1e4).max(dim=-1).values
            losses["new_existing_margin"].append(0.75 * F.softplus(best_state - output["new_logit"] + 0.2).mean())
        if desired == "DEFER":
            losses["commit_defer_margin"].append(0.75 * F.softplus(output["new_logit"] - output["defer_logit"] + 0.2).mean())
        elif desired in {"EXISTING", "NEW", "KNOWN"}:
            if desired == "EXISTING" and desired_slot is not None:
                correct_logit = output["state_logits"][:, int(desired_slot)]
            elif desired == "KNOWN" and desired_known is not None:
                correct_logit = output["known_logits"][:, int(desired_known)]
            else:
                correct_logit = output["new_logit"]
            losses["commit_defer_margin"].append(0.75 * F.softplus(output["defer_logit"] - correct_logit + 0.2).mean())
        if desired == "RESET":
            non_reset = torch.cat([output["new_logit"].reshape(1, 1), output["defer_logit"].reshape(1, 1), output["state_logits"].masked_fill(~valid, -1e4)], dim=1).max(dim=-1).values
            losses["reset_margin"].append(0.75 * F.softplus(non_reset - output["reset_logit"] + 0.2).mean())
        with torch.no_grad():
            predicted, pred_slot, confidence = decode_joint_action(logits, model.known_count, model.max_states)
        use_teacher = bool(train and rng.random() < teacher_probability)
        if use_teacher:
            chosen = desired
            chosen_slot = desired_slot if desired == "EXISTING" else None
            chosen_known = desired_known if desired == "KNOWN" else None
            chosen_global = desired_global if desired == "EXISTING" else None
        else:
            chosen = predicted
            chosen_slot = pred_slot if predicted == "EXISTING" else None
            chosen_known = pred_slot if predicted == "KNOWN" else None
            chosen_global = (
                tensors["state_indices"][pred_slot]
                if predicted == "EXISTING" and pred_slot is not None
                and pred_slot < len(tensors["state_indices"])
                else None
            )
        if chosen == "RESET":
            if session.reset_injection_applied:
                session.reset_reason = "synthetic_wrong_binding"
            elif session.committed_action == "NEW":
                session.reset_reason = "wrong_new"
            elif session.committed_action is None and position + 1 < reliable:
                session.reset_reason = "premature_commit"
            else:
                session.reset_reason = "wrong_existing"
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
        session.collected_vectors.append(
            encoded["semantic_seq"][0, position].detach().cpu().numpy().astype(np.float32, copy=True)
        )
        trace.append({
            "track_key": track_key, "role": role, "position": position + 1,
            "target": desired, "predicted": predicted, "chosen": chosen,
            "use_teacher": use_teacher, "confidence": confidence,
            "target_index": target_index, "state_slot": desired_slot,
            "chosen_state_slot": chosen_slot,
            "chosen_global_state": chosen_global,
            "chosen_known": chosen_known,
            "reset": int(chosen == "RESET"),
            "reset_reason": session.reset_reason,
            "missing_positive_state": bool(session.missing_positive_state),
            "collected_count": len(session.collected_vectors),
            "best_streak": int(session.best_streak),
        })
    if session.collected_vectors:
        final_np = np.mean(np.asarray(session.collected_vectors, dtype=np.float32), axis=0)
        final_vector = torch.from_numpy(final_np).to(device)
    else:
        final_vector = encoded["semantic"].squeeze(0)
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
    target_trace = [x for x in trace if x.get("role") == "target"]
    source_trace = [x for x in trace if x.get("role") == "source"]
    target_counts = {name: sum(int(x.get("target") == name) for x in target_trace) for name in ("EXISTING", "NEW", "DEFER", "RESET")}
    source_counts = {name: sum(int(x.get("target") == name) for x in source_trace) for name in ("EXISTING", "NEW", "DEFER", "RESET")}
    reset_reasons = {}
    for x in trace:
        reason = x.get("reset_reason")
        if x.get("target") == "RESET" or x.get("chosen") == "RESET":
            reset_reasons[reason or "unknown"] = reset_reasons.get(reason or "unknown", 0) + 1
    return total, {
        "event_id": event.get("event_id"), "polarity": event.get("polarity"), "trace": trace,
        "losses": {k: float(v.detach().cpu()) for k, v in present.items()},
        "state_count": len(memory.states),
        "reset_targets": sum(int(x["target"] == "RESET") for x in trace),
        "reset_predictions": sum(int(x["chosen"] == "RESET") for x in trace),
        "reset_targets_source": int(source_counts["RESET"]),
        "reset_targets_target": int(target_counts["RESET"]),
        "source_target_counts": source_counts,
        "target_action_counts": target_counts,
        "masked_target_violations": 0,
        "reset_reason_counts": reset_reasons,
    }
