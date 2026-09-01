"""Persistent held-known event evaluator for Phase19R."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.models.known_osr import GaussianController, RawPersistentController, TALONStyleController
from src.iclr27_phase19r.runtime.runner import ModelStreamController


def load_events(fold: int | None = None) -> list[dict[str, Any]]:
    root = Path("outputs/iclr27_phase19r/manifests")
    out = [json.loads(x) for p in [root / "held_known_positive_events.jsonl", root / "held_known_negative_events.jsonl"] for x in p.read_text().splitlines() if x.strip()]
    return [e for e in out if fold is None or int(e["fold"]) == int(fold)]


def _process_model(ctrl: ModelStreamController, data: Phase19RData, key: str, cat: int, known_mask: torch.Tensor, device: torch.device, phase: str) -> list[dict[str, Any]]:
    out = []
    for pos in range(len(data.track_rows[key])):
        raw, geom, quality, _ = data.prefix(key, pos); row = data.rows[data.track_rows[key][pos]]
        rec = ctrl.process_item(torch.from_numpy(raw).to(device), torch.from_numpy(geom).to(device), quality, int(row["video_id"]), key, known_mask, oracle_category=cat)
        out.append({"row_key": row["row_key"], "position": pos, "phase": phase, "action": rec["action"], "semantic_id": rec.get("semantic_id"), "quality": quality, "confidence": rec.get("selected_confidence", 0.), "state_count": rec["state_count"]})
    return out


def simulate(controller: Any, data: Phase19RData, event: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if hasattr(controller, "reset_stream"): controller.reset_stream()
    target_cat = int(event.get("target_category_gt_denominator_only", event.get("category_gt_denominator_only")))
    source_cat = int(event.get("distractor_category_gt_denominator_only", event.get("category_gt_denominator_only")))
    if isinstance(controller, ModelStreamController):
        # Keep the episode-conditioned mask on the same device as model logits.
        mask_np = np.asarray(data.active_known_mask, dtype=bool).copy()
        # Pseudo-held training events may hide supported TRAIN categories to
        # emulate a held novel class.  This is an evaluator-side mask only;
        # public/held manifests do not contain this field and retain the
        # registered fold mask unchanged.
        for cat in event.get("masked_known_categories", []):
            j = data.known_to_index.get(int(cat))
            if j is not None:
                mask_np[j] = False
        km = torch.from_numpy(mask_np).to(device)
        source = [_process_model(controller, data, k, source_cat, km, device, "source") for k in event["source_tracklet_keys"]]
        target = _process_model(controller, data, event["target_tracklet_key"], target_cat, km, device, "target")
        states = {s.sid: s for s in controller.memory.states}
    else:
        source = [controller.process_track(k, phase="source", eval_category=source_cat) for k in event["source_tracklet_keys"]]
        target = controller.process_track(event["target_tracklet_key"], phase="target", eval_category=target_cat)
        states = {s.sid: s for s in controller.memory.states}
    prefix = int(event["target_first_reliable_prefix_index_gt_only"]); post = target[prefix:]
    def correct(d: dict[str, Any]) -> bool:
        if d.get("action") != "EXISTING" or d.get("semantic_id") is None: return False
        s = states.get(int(d["semantic_id"])); birth_cat = getattr(s, "oracle_birth_category", None)
        return bool(s and birth_cat == target_cat and int(s.birth_video) != int(event["target_video"]) and str(s.birth_track) != str(event["target_tracklet_key"]))
    first = next((d for d in post if d["action"] != "DEFER"), None)
    existing = [d for d in post if d["action"] == "EXISTING"]
    return {"event_key": event["event_key"], "kind": event["kind"], "fold": int(event["fold"]), "target_category": target_cat, "target_video": int(event["target_video"]), "source_decisions": [x for y in source for x in y], "target_decisions": target, "first_commit": first, "first_commit_correct": bool(first and correct(first)), "post_prefix_correct_rows": int(sum(correct(d) for d in post)), "post_prefix_rows": len(post), "existing_correct_rows": int(sum(correct(d) for d in existing)), "existing_rows": len(existing), "negative_false_merge": bool(event["kind"] == "negative_new" and first and first["action"] == "EXISTING"), "pre_prefix_defer_rows": sum(d["action"] == "DEFER" for d in target[:prefix]), "pre_prefix_rows": prefix, "premature": any(d["action"] != "DEFER" for d in target[:prefix]), "unresolved": first is None, "state_count": len(states), "duplicate_target_births": sum(getattr(s, "oracle_birth_category", None) == target_cat and int(getattr(s, "birth_video", -1)) == int(event["target_video"]) for s in states.values()), "states": [s.snapshot() for s in states.values()]}


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def metrics(records: list[dict[str, Any]], known_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the preregistered event-level and category-macro terms."""
    pos = [r for r in records if r["kind"] == "positive_existing"]; neg = [r for r in records if r["kind"] == "negative_new"]
    by_cat: dict[int, list[int]] = defaultdict(list); by_video: dict[int, list[int]] = defaultdict(list)
    for r in pos:
        by_cat[int(r["target_category"])].append(int(r["first_commit_correct"])); by_video[int(r["target_video"])].append(int(r["first_commit_correct"]))
    ex_total = sum(r["existing_rows"] for r in records); ex_good = sum(r["existing_correct_rows"] for r in records); pp_total = sum(r["post_prefix_rows"] for r in pos); pp_good = sum(r["post_prefix_correct_rows"] for r in pos)
    recall = float(np.mean([r["first_commit_correct"] for r in pos])) if pos else 0.
    ex_p = ex_good / max(ex_total, 1); ex_r = ex_good / max(pp_total, 1)
    first_by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in pos + neg: first_by_cat[int(r["target_category"])].append(r)
    ex_f1_c = []; new_f1_c = []; false_c = []
    for cat, rs in sorted(first_by_cat.items()):
        pc = [r for r in rs if r["kind"] == "positive_existing"]; nc = [r for r in rs if r["kind"] == "negative_new"]
        tp_ex = sum(bool(r.get("first_commit") and r["first_commit"].get("action") == "EXISTING" and r["first_commit_correct"]) for r in pc)
        fp_ex = sum(bool(r.get("first_commit") and r["first_commit"].get("action") == "EXISTING" and not r["first_commit_correct"]) for r in pc + nc)
        ex_f1_c.append(_f1(tp_ex, fp_ex, len(pc) - tp_ex))
        tp_new = sum(bool(r.get("first_commit") and r["first_commit"].get("action") == "NEW") for r in nc)
        fp_new = sum(bool(r.get("first_commit") and r["first_commit"].get("action") == "NEW") for r in pc)
        new_f1_c.append(_f1(tp_new, fp_new, len(nc) - tp_new))
        false_c.append(float(np.mean([r["negative_false_merge"] for r in nc])) if nc else 0.0)
    # NMI is computed on the anonymous SID selected at the first post-prefix
    # commit.  Missing/deferred commits map to -1, preserving the denominator.
    nmi_labels = []; nmi_pred = []
    for r in pos:
        nmi_labels.append(int(r["target_category"])); fc = r.get("first_commit")
        nmi_pred.append(int(fc["semantic_id"]) if fc and fc.get("semantic_id") is not None else -1)
    nmi = float(normalized_mutual_info_score(nmi_labels, nmi_pred)) if len(set(nmi_labels)) > 1 else 0.0
    known_metrics = known_metrics or {}
    known_macro = float(known_metrics.get("known_macro", 0.0)); known_micro = float(known_metrics.get("known_micro", 0.0))
    existing_f1_macro = float(np.mean(ex_f1_c)) if ex_f1_c else 0.0
    new_f1_macro = float(np.mean(new_f1_c)) if new_f1_c else 0.0
    reuse_macro = float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0
    false_macro = float(np.mean(false_c)) if false_c else 0.0
    frag_macro = float(sum(r["duplicate_target_births"] for r in records) / max(len(records), 1))
    score = (.30 * existing_f1_macro + .20 * new_f1_macro + .15 * known_macro
             + .15 * reuse_macro + .10 * nmi - .10 * false_macro - .05 * frag_macro)
    return {"positive_events": len(pos), "negative_events": len(neg), "commit_ct": {"correct": int(sum(r["first_commit_correct"] for r in pos)), "eligible": len(pos), "recall": recall}, "post_prefix_ct": {"correct_rows": pp_good, "rows": pp_total, "recall": pp_good / max(pp_total, 1)}, "existing_precision": ex_p, "existing_recall": ex_r, "existing_f1": 2*ex_p*ex_r/max(ex_p+ex_r,1e-9), "existing_f1_macro": existing_f1_macro, "new_precision": sum(bool(r.get("first_commit") and r["first_commit"].get("action") == "NEW") for r in neg) / max(sum(bool(r.get("first_commit") and r["first_commit"].get("action") == "NEW") for r in pos + neg), 1), "new_recall": sum(bool(r.get("first_commit") and r["first_commit"].get("action") == "NEW") for r in neg) / max(len(neg), 1), "new_f1_macro": new_f1_macro, "known_macro": known_macro, "known_micro": known_micro, "positive_reuse_recall_macro": reuse_macro, "novel_nmi_macro": nmi, "negative_false_merge_rate": float(np.mean([r["negative_false_merge"] for r in neg])) if neg else 0., "false_merge_rate_macro": false_macro, "fragmentation_rate_macro": frag_macro, "selection_score": float(score), "duplicate_births": int(sum(r["duplicate_target_births"] for r in records)), "premature_rate": float(np.mean([r["premature"] for r in records])) if records else 0., "unresolved_rate": float(np.mean([r["unresolved"] for r in records])) if records else 0., "pre_prefix_defer_rate": sum(r["pre_prefix_defer_rows"] for r in records)/max(sum(r["pre_prefix_rows"] for r in records),1), "category_macro_reuse": reuse_macro, "category_coverage": sum(any(v) for v in by_cat.values()), "video_coverage": sum(any(v) for v in by_video.values()), "by_category": {str(k): {"correct": int(sum(v)), "eligible": len(v), "recall": float(np.mean(v))} for k,v in sorted(by_cat.items())}}


def evaluate_candidate(name: str, data: Phase19RData, checkpoint: Path | None, device: torch.device, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if name == "raw": ctrl: Any = RawPersistentController(data, deferred=True)
    elif name in {"age", "fallback"}: ctrl = GaussianController(data, deferred=True)
    elif name == "talon": ctrl = TALONStyleController(data, deferred=True)
    else:
        if checkpoint is None: raise ValueError("checkpoint required")
        ck = torch.load(checkpoint, map_location="cpu"); model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias)); model.load_state_dict(ck["model_state"]); model.to(device).eval(); ctrl = ModelStreamController(model, max_states=16, allow_defer=True, tau_ready=model.tau_ready, tau_known=model.tau_known, tau_assign=model.tau_assign)
    events = events or load_events(data.fold)
    with torch.no_grad():
        records = [simulate(ctrl, data, e, device) for e in events]
        known_val = evaluate_known_stream(ctrl, data, device, fixed_known_keys(data)) if isinstance(ctrl, ModelStreamController) else {}
    return {"protocol": "trackocd_iclr27_phase19r_persistent_internal_events", "candidate": name, "fold": data.fold, "events": len(records), "metrics": metrics(records, known_val), "known_metrics": known_val, "records": records}


def evaluate_model_instance(model: RCMSOCD, data: Phase19RData, device: torch.device,
                            events: list[dict[str, Any]] | None = None,
                            known_keys: list[tuple[str, int]] | None = None) -> dict[str, Any]:
    """Evaluate an in-memory model with the same persistent event replay used for checkpoints.

    This function deliberately shares ``simulate`` and ``metrics`` with the offline evaluator.
    It is used during training so checkpoint selection cannot silently fall back to a different
    three-step/episode proxy.  The model is never updated and every event gets a fresh stream
    controller, preserving causal state isolation.
    """
    model.eval()
    ctrl = ModelStreamController(model, max_states=16, allow_defer=True,
                                 tau_ready=model.tau_ready, tau_known=model.tau_known,
                                 tau_assign=model.tau_assign)
    evs = events or load_events(data.fold)
    with torch.no_grad():
        records = [simulate(ctrl, data, e, device) for e in evs]
        known_val = evaluate_known_stream(ctrl, data, device, known_keys or [])
    model.train()
    return {"protocol": "trackocd_iclr27_phase19r_persistent_internal_events",
            "candidate": "main_in_memory", "fold": data.fold,
            "events": len(records), "metrics": metrics(records, known_val), "known_metrics": known_val,
            "records": records}


def fixed_known_keys(data: Phase19RData) -> list[tuple[str, int]]:
    """Deterministic legal supported-known streams for checkpoint safety."""
    out: list[tuple[str, int]] = []
    for cat in sorted(data.train_categories):
        choices = [k for k in data.category_tracks.get(cat, []) if data.track_video[k] in data.fit_videos]
        if choices:
            out.append((sorted(choices)[0], int(cat)))
    return out


def evaluate_known_stream(controller: ModelStreamController, data: Phase19RData,
                          device: torch.device, keys: list[tuple[str, int]]) -> dict[str, Any]:
    by_cat: dict[int, list[int]] = defaultdict(list); rows = 0
    km = torch.from_numpy(data.active_known_mask).to(device)
    for key, cat in keys:
        controller.reset_stream()
        for pos in range(len(data.track_rows[key])):
            raw, geom, quality, _ = data.prefix(key, pos); row = data.rows[data.track_rows[key][pos]]
            got = controller.process_item(torch.from_numpy(raw).to(device), torch.from_numpy(geom).to(device), quality,
                                          int(row["video_id"]), key, km, oracle_category=None)
            if quality >= controller.tau_ready:
                by_cat[int(cat)].append(int(got["action"] == "KNOWN")); rows += 1
    vals = [x for v in by_cat.values() for x in v]
    return {"known_micro": float(np.mean(vals)) if vals else 0.0,
            "known_macro": float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0,
            "known_rows": rows, "known_categories": len(by_cat),
            "by_category": {str(c): {"rows": len(v), "accuracy": float(np.mean(v))} for c, v in sorted(by_cat.items())}}
