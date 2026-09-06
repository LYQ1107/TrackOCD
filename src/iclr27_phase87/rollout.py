"""The train and evaluation event rollout core."""
from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F

from src.iclr27_phase87.controller import CausalPersistentOCD, mask_joint_logits
from src.iclr27_phase87.data import FeatureStore, pad_track


def _event_tensors(store: FeatureStore, event: dict[str, Any], device: torch.device):
    target = store.track(event["target_track_key"])
    source = store.track(event["source_track_keys"][0])
    traw, tgeom, tquality, tmask = pad_track(target)
    sraw, sgeom, _, smask = pad_track(source)
    return tuple(torch.from_numpy(value).unsqueeze(0).to(device) for value in (sraw, sgeom, smask, traw, tgeom, tquality, tmask)), target


def rollout_event(model: CausalPersistentOCD, store: FeatureStore, event: dict[str, Any], train: bool, teacher_probability: float = 0.0, rng: random.Random | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    device = next(model.parameters()).device
    tensors, target_array = _event_tensors(store, event, device)
    sraw, sgeom, smask, traw, tgeom, tquality, tmask = tensors
    source_encoded = model.encode_track(sraw, sgeom, smask)
    target_encoded = model.encode_track(traw, tgeom, tmask)
    prototypes = torch.zeros((1, model.max_states, 4, 768), device=device)
    prototype_mask = torch.zeros((1, model.max_states, 4), dtype=torch.bool, device=device)
    state_stats = torch.zeros((1, model.max_states, 6), device=device)
    prototypes[:, 0, 0] = source_encoded["semantic"].detach() if not train else source_encoded["semantic"]
    prototype_mask[:, 0, 0] = True
    state_stats[:, 0, :5] = torch.tensor([1.0 / 32.0, min(len(target_array.raw), 16) / 64.0, min(len(target_array.raw), 16) / 32.0, 0.0, 0.25], device=device)
    support = torch.zeros((1, 8), device=device)
    previous = torch.zeros((1, model.max_states), device=device)
    committed_action: str | None = None
    committed_state: int | None = None
    evidence = previous
    loss_terms = {"action_ce": [], "state_relation": [], "false_merge_risk": [], "new_existing_margin": [], "commit_defer_margin": [], "reset_margin": []}
    traces = []
    reliable = int(event.get("reliable_prefix_for_loss_only", min(len(target_array.raw), 16)))
    positive = event.get("polarity") == "positive"
    for position in range(min(len(target_array.raw), 16)):
        output = model.forward_action(target_encoded["semantic_seq"][:, position], target_encoded["track_hidden_seq"][:, position], prototypes, prototype_mask, state_stats, evidence, support, tquality[:, position], torch.tensor([0.0], device=device))
        logits = mask_joint_logits(output["joint_logits"], model.known_count, model.max_states, [committed_action], [committed_state])
        if position + 1 < reliable:
            target_index = model.known_count + model.max_states + 1
            target_name = "DEFER"
        elif positive:
            target_index = model.known_count
            target_name = "EXISTING"
        else:
            target_index = model.known_count + model.max_states
            target_name = "NEW"
        loss_terms["action_ce"].append(F.cross_entropy(logits, torch.tensor([target_index], device=device)))
        relation_target = torch.tensor([[1.0 if positive else 0.0] + [0.0] * (model.max_states - 1)], device=device)
        relation_target = relation_target[:, : output["state_logits"].shape[1]]
        loss_terms["state_relation"].append(F.binary_cross_entropy_with_logits(output["state_logits"], relation_target))
        if not positive:
            loss_terms["false_merge_risk"].append(2.0 * F.softplus(output["state_logits"][:, 0]).mean())
        if positive:
            loss_terms["new_existing_margin"].append(0.75 * F.softplus(output["new_logit"] - output["state_logits"][:, 0] + 0.2).mean())
        else:
            loss_terms["new_existing_margin"].append(0.75 * F.softplus(output["state_logits"][:, 0] - output["new_logit"] + 0.2).mean())
        correct_commit = output["state_logits"][:, 0] if positive else output["new_logit"]
        if position + 1 < reliable:
            loss_terms["commit_defer_margin"].append(0.75 * F.softplus(correct_commit - output["defer_logit"] + 0.2).mean())
        else:
            loss_terms["commit_defer_margin"].append(0.75 * F.softplus(output["defer_logit"] - correct_commit + 0.2).mean())
        with torch.no_grad():
            predicted = int(torch.argmax(logits, dim=-1).item())
        use_teacher = train and rng is not None and rng.random() < teacher_probability
        chosen = target_name if use_teacher else ("EXISTING" if predicted < model.known_count + model.max_states else ("NEW" if predicted == model.known_count + model.max_states else ("DEFER" if predicted == model.known_count + model.max_states + 1 else "RESET")))
        if committed_action is None and chosen in {"EXISTING", "NEW"}:
            committed_action = chosen
            committed_state = 0 if chosen == "EXISTING" else None
        elif chosen == "RESET":
            committed_action = None
            committed_state = None
            evidence = torch.zeros_like(evidence)
        evidence = 0.70 * evidence + 0.30 * torch.sigmoid(output["state_logits"])
        traces.append({"position": position + 1, "target": target_name, "predicted": chosen, "target_index": target_index, "new_logit": float(output["new_logit"].item()), "defer_logit": float(output["defer_logit"].item()), "reset_logit": float(output["reset_logit"].item()), "best_state_logit": float(output["state_logits"][:, 0].item()), "evidence": evidence.detach().cpu().tolist()})
    weights = {"action_ce": 1.0, "state_relation": 1.0, "false_merge_risk": 1.0, "new_existing_margin": 1.0, "commit_defer_margin": 1.0, "reset_margin": 1.0}
    total = sum(weights[name] * (sum(values) / max(1, len(values))) for name, values in loss_terms.items() if values)
    return total, {"event_id": event.get("event_id"), "polarity": event.get("polarity"), "target": event.get("target_track_key"), "trace": traces, "losses": {name: float(sum(values).detach().cpu() / max(1, len(values))) for name, values in loss_terms.items() if values}, "final_committed_action": committed_action}
