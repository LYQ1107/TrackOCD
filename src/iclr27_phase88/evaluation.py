"""Phase88 evaluator adapter using the exact Phase19R metric implementation."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch

from src.iclr27_phase19r.evaluation.internal import fixed_known_keys, metrics as phase19r_metrics
from src.iclr27_phase88.controller import CausalPersistentOCD
from src.iclr27_phase88.data import FeatureStore
from src.iclr27_phase88.runtime import CausalPersistentRuntime


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    out = dict(event)
    if "target_track_key" not in out:
        out["target_track_key"] = out.get("target_tracklet_key")
    if "source_track_keys" not in out:
        out["source_track_keys"] = list(out.get("source_tracklet_keys", []))
    if "polarity" not in out:
        out["polarity"] = "positive" if out.get("kind") == "positive_existing" else "negative"
    if "target_category_for_loss_only" not in out:
        out["target_category_for_loss_only"] = out.get("category_gt_denominator_only", out.get("target_category_gt_denominator_only"))
    if "target_video" not in out:
        out["target_video"] = out.get("target_video", -1)
    if "reliable_prefix_for_loss_only" not in out:
        out["reliable_prefix_for_loss_only"] = int(out.get("target_first_reliable_prefix_index_gt_only", 0)) + 1
    if "masked_known_categories_for_loss_only" not in out:
        out["masked_known_categories_for_loss_only"] = list(out.get("masked_known_categories", []))
    return out


def _known_mask(data, event: dict[str, Any], device: torch.device) -> torch.Tensor:
    mask = np.asarray(data.active_known_mask, dtype=bool).copy()
    for cat in event.get("masked_known_categories_for_loss_only", []):
        j = data.known_to_index.get(int(cat))
        if j is not None:
            mask[j] = False
    return torch.from_numpy(mask).to(device)


def _correct_existing(row: dict[str, Any], states: dict[int, dict[str, Any]], target_category: int,
                      target_video: int, target_track: str) -> bool:
    if row.get("action") != "EXISTING" or row.get("semantic_id") is None:
        return False
    state = states.get(int(row["semantic_id"]))
    return bool(
        state and state.get("oracle_birth_category") == int(target_category)
        and int(state.get("birth_video", -1)) != int(target_video)
        and str(state.get("birth_track", "")) != str(target_track)
    )


def replay_persistent_records(model: CausalPersistentOCD, data, events: list[dict[str, Any]],
                              device: torch.device, *, support_mode: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay source/target events only; known-stream evaluation is separate."""
    model.eval()
    store = FeatureStore(data.fold, data)
    records: list[dict[str, Any]] = []
    diagnostic = Counter()
    with torch.no_grad():
        for raw_event in events:
            event = normalize_event(raw_event)
            runtime = CausalPersistentRuntime(model, store, device)
            runtime.reset_stream()
            km = _known_mask(data, event, device)
            source_decisions = []
            source_created = 0
            for source_key in event["source_track_keys"]:
                source_cat = int(data.category(source_key) if hasattr(data, "category") else data.track_category.get(source_key, -1))
                source_video = int(data.video(source_key) if hasattr(data, "video") else data.track_video[source_key])
                result = runtime.process_track(source_key, source_video, km,
                                               oracle_category_for_eval=source_cat,
                                               support_mode=support_mode)
                source_decisions.extend(result["trace"])
                source_created += int(result["final_action"] == "NEW")
            target_key = event["target_track_key"]
            target_cat = int(event["target_category_for_loss_only"])
            target_video = int(event["target_video"])
            target_result = runtime.process_track(target_key, target_video, km,
                                                  oracle_category_for_eval=target_cat,
                                                  support_mode=support_mode)
            target_decisions = target_result["trace"]
            states = {int(s["sid"]): s for s in runtime.memory.snapshot()}
            reliable = max(1, min(16, int(event.get("reliable_prefix_for_loss_only", 1))))
            pre = target_decisions[: reliable - 1]
            post = target_decisions[reliable - 1 :]
            first = next((row for row in post if row.get("action") in {"KNOWN", "EXISTING", "NEW"}), None)
            pre_commits = [row for row in pre if row.get("action") in {"KNOWN", "EXISTING", "NEW"}]
            is_positive = event.get("polarity") == "positive"
            correct = bool(first and _correct_existing(first, states, target_cat, target_video, target_key)) if is_positive else False
            negative_merge = bool((not is_positive) and first and first.get("action") in {"EXISTING", "KNOWN"})
            duplicate_births = sum(
                s.get("oracle_birth_category") == target_cat and int(s.get("birth_video", -1)) == target_video
                for s in states.values()
            )
            record = {
                "event_key": event.get("event_key", event.get("event_id")),
                "kind": "positive_existing" if is_positive else "negative_new",
                "fold": int(event.get("fold", data.fold)),
                "target_category": target_cat,
                "target_video": target_video,
                "source_decisions": source_decisions,
                "target_decisions": target_decisions,
                "first_commit": first,
                "first_commit_correct": correct,
                "post_prefix_correct_rows": int(sum(_correct_existing(x, states, target_cat, target_video, target_key) for x in post)),
                "post_prefix_rows": len(post),
                "existing_correct_rows": int(sum(_correct_existing(x, states, target_cat, target_video, target_key) for x in post if x.get("action") == "EXISTING")),
                "existing_rows": int(sum(x.get("action") == "EXISTING" for x in post)),
                "negative_false_merge": negative_merge,
                "pre_prefix_defer_rows": int(sum(x.get("action") == "DEFER" for x in pre)),
                "pre_prefix_rows": len(pre),
                "premature": bool(pre_commits),
                "unresolved": first is None,
                "state_count": len(states),
                "duplicate_target_births": int(duplicate_births),
                "states": list(states.values()),
            }
            records.append(record)
            diagnostic.update({
                "source_tracks_processed": len(event["source_track_keys"]),
                "source_states_created": source_created,
                "source_defer": sum(x.get("action") == "DEFER" for x in source_decisions),
                "target_candidates_final": sum(x.get("action") == "EXISTING" for x in target_decisions),
                "reset_count": sum(x.get("reset", 0) for x in target_decisions),
                "premature": int(bool(pre_commits)),
            })
    diagnostic_out = {
            "source_tracks_processed": int(diagnostic["source_tracks_processed"]),
            "source_states_created": int(diagnostic["source_states_created"]),
            "source_deferred": int(diagnostic["source_defer"]),
            "target_candidate_existing_rows": int(diagnostic["target_candidates_final"]),
            "reset_count": int(diagnostic["reset_count"]),
            "premature_count": int(diagnostic["premature"]),
            "mean_states_per_event": float(np.mean([r["state_count"] for r in records])) if records else 0.0,
            "mean_prototypes_per_state": float(np.mean([s["prototype_count"] for r in records for s in r["states"]])) if any(r["states"] for r in records) else 0.0,
            "multi_prototype_state_fraction": float(np.mean([s["prototype_count"] > 1 for r in records for s in r["states"]])) if any(r["states"] for r in records) else 0.0,
            "state_impurity": int(sum(s.get("impurity_count", 0) for r in records for s in r["states"])),
            "reset_targets": 0,
            "reset_predictions": int(diagnostic["reset_count"]),
            "source_track_count": int(sum(len(normalize_event(e)["source_track_keys"]) for e in events)),
        }
    return records, diagnostic_out


def finalize_persistent_metrics(records: list[dict[str, Any]], known_metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the exact Phase19R metrics implementation after all shards."""
    return phase19r_metrics(records, known_metrics)


def evaluate_persistent_events(model: CausalPersistentOCD, data, events: list[dict[str, Any]],
                               device: torch.device, *, support_mode: bool = False) -> dict[str, Any]:
    """Compatibility wrapper: event replay plus one known-stream pass."""
    records, diagnostic = replay_persistent_records(model, data, events, device, support_mode=support_mode)
    known = evaluate_known_stream_v2(model, data, device)
    return {"records": records, "known_metrics": known,
            "metrics": finalize_persistent_metrics(records, known), "diagnostic": diagnostic}


def evaluate_known_stream_v2(model: CausalPersistentOCD, data, device: torch.device) -> dict[str, Any]:
    """Evaluate fit known streams using the same runtime and known buffers."""
    store = FeatureStore(data.fold, data)
    by_cat: dict[int, list[int]] = {}
    rows = 0
    km = torch.from_numpy(np.asarray(data.active_known_mask, dtype=bool)).to(device)
    with torch.no_grad():
        known_keys = data.known_eval_keys if hasattr(data, "known_eval_keys") else fixed_known_keys(data)
        for key, cat in known_keys:
            runtime = CausalPersistentRuntime(model, store, device)
            key_video = int(data.video(key) if hasattr(data, "video") else data.track_video[key])
            result = runtime.process_track(key, key_video, km)
            values = []
            slot = data.known_to_index.get(int(cat))
            for row in result["trace"]:
                if float(row.get("quality", 0.0)) >= 0.45:
                    values.append(int(row.get("action") == "KNOWN" and row.get("known_index") == slot))
            if values:
                by_cat[int(cat)] = values
                rows += len(values)
    vals = [x for v in by_cat.values() for x in v]
    return {
        "known_micro": float(np.mean(vals)) if vals else 0.0,
        "known_macro": float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0,
        "known_rows": int(rows),
        "known_categories": len(by_cat),
        "by_category": {str(c): {"rows": len(v), "accuracy": float(np.mean(v))} for c, v in sorted(by_cat.items())},
    }
