"""Causal DSTM state machine, calibration, and fixed-event evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.iclr27_phase18.evaluation.baseline_b1 import event_metrics, threshold_curve
from src.iclr27_phase18.models.dstm import DSTM
from src.iclr27_phase18.training.data import FoldData, load_jsonl, OUT


def build_category_events(data: FoldData, categories: list[int]) -> list[dict[str, Any]]:
    cat_tracks: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for key, t in data.track_manifest.items():
        c = int(t["label_category_gt_only"])
        if c in categories and int(t["first_reliable_prefix_index_gt_only"]) >= 0:
            cat_tracks[c][int(t["video_id"])].append(key)
    for videos in cat_tracks.values():
        for v in videos:
            videos[v].sort()
    positives = []
    for c in sorted(categories):
        videos = cat_tracks[c]
        for sv in sorted(videos):
            for tv in sorted(videos):
                if sv == tv:
                    continue
                for target_key in videos[tv]:
                    t = data.track_manifest[target_key]
                    positives.append({
                        "event_key": f"calpos:c{c}:sv{sv}:tv{tv}:tt{target_key}",
                        "kind": "positive_existing", "fold": data.fold_id,
                        "category_gt_denominator_only": c, "source_video": sv,
                        "target_video": tv, "source_tracklet_keys": videos[sv],
                        "target_tracklet_key": target_key,
                        "target_first_reliable_prefix_index_gt_only": int(t["first_reliable_prefix_index_gt_only"]),
                    })
    negatives = []
    cats = sorted(categories)
    for p in positives:
        c = int(p["category_gt_denominator_only"]); dc = cats[(cats.index(c) + 1) % len(cats)]
        candidate_videos = [v for v in sorted(cat_tracks[dc]) if v != p["target_video"]]
        if not candidate_videos:
            candidate_videos = sorted(cat_tracks[dc])
        dv = candidate_videos[0]
        negatives.append({
            "event_key": p["event_key"].replace("calpos:", "calneg:", 1),
            "kind": "negative_new", "fold": data.fold_id,
            "target_category_gt_denominator_only": c,
            "distractor_category_gt_denominator_only": dc,
            "source_video": dv, "target_video": p["target_video"],
            "source_tracklet_keys": cat_tracks[dc][dv],
            "target_tracklet_key": p["target_tracklet_key"],
            "target_first_reliable_prefix_index_gt_only": p["target_first_reliable_prefix_index_gt_only"],
        })
    return sorted(positives + negatives, key=lambda x: x["event_key"])


class Runtime:
    def __init__(self, model: DSTM, data: FoldData, device: torch.device,
                 variant: str, tau_ready: float | None = None,
                 exact_readiness: bool = False):
        self.model = model.eval(); self.data = data; self.device = device
        self.variant = variant; self.tau_ready = tau_ready
        self.exact_readiness = exact_readiness
        self.cache: dict[str, dict[str, np.ndarray]] = {}

    @torch.no_grad()
    def track_outputs(self, track_key: str) -> dict[str, np.ndarray]:
        if track_key in self.cache:
            return self.cache[track_key]
        indices = [int(x) for x in self.data.track_manifest[track_key]["row_indices"]]
        max_len = int(self.data.config["model"]["max_causal_sequence_rows"])
        batch_size = 256; tokens, scores, known = [], [], []
        for start in range(0, len(indices), batch_size):
            positions = list(range(start, min(start + batch_size, len(indices))))
            q = np.zeros((len(positions), max_len, self.data.input_dim), np.float16)
            lengths = np.zeros(len(positions), np.int64)
            for j, pos in enumerate(positions):
                seq = indices[max(0, pos + 1 - max_len):pos + 1]
                q[j, :len(seq)] = self.data.row_input[seq]; lengths[j] = len(seq)
            qt = torch.from_numpy(q).to(self.device, dtype=torch.float32)
            lt = torch.from_numpy(lengths).to(self.device)
            token, _ = self.model.encode_sequence(qt, lt)
            tokens.append(token.float().cpu().numpy())
            scores.append(torch.sigmoid(self.model.reliability_head(token).squeeze(-1)).float().cpu().numpy())
            known.append(self.model.known_aux(token).float().cpu().numpy())
        value = {"indices": np.asarray(indices, np.int64), "tokens": np.concatenate(tokens),
                 "readiness": np.concatenate(scores), "known_aux": np.concatenate(known)}
        self.cache[track_key] = value
        return value

    @torch.no_grad()
    def decode(self, query: np.ndarray, memory: dict[int, dict[str, Any]],
               track_key: str, video: int) -> tuple[str, int | None, float, dict[str, Any]]:
        candidates = [(sid, state) for sid, state in sorted(memory.items())
                      if state["birth_tracklet"] != track_key and state["birth_video"] != video]
        summaries = []
        for _, state in candidates:
            order = np.argsort(-np.asarray(state["anchor_scores"], np.float32))[:8]
            summaries.append(np.mean(np.asarray(state["anchors"], np.float32)[order], axis=0))
        if summaries:
            state_rows = torch.from_numpy(np.asarray(summaries, np.float32))[None].to(self.device)
            state_mask = torch.ones(1, len(summaries), dtype=torch.bool, device=self.device)
        else:
            state_rows = torch.zeros(1, 0, self.data.input_dim, device=self.device)
            state_mask = torch.zeros(1, 0, dtype=torch.bool, device=self.device)
        q = torch.from_numpy(query.astype(np.float32))[None].to(self.device)
        state_tokens = self.model.encode_state(state_rows)
        allow_defer = self.variant not in {"b3", "repair_r1"}
        logits, _ = self.model.decode(q, state_tokens, state_mask, allow_defer=allow_defer)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs)); confidence = float(probs[idx])
        known_n = len(self.data.known_ids); state_n = len(candidates)
        if idx < known_n:
            return "KNOWN", int(self.data.known_ids[idx]), confidence, {"candidate_states": state_n}
        if idx < known_n + state_n:
            return "EXISTING_NOVEL", int(candidates[idx - known_n][0]), confidence, {"candidate_states": state_n}
        if idx == known_n + state_n:
            return "NEW_NOVEL", None, confidence, {"candidate_states": state_n}
        return "DEFER", None, confidence, {"candidate_states": state_n}

    def simulate_event(self, event: dict[str, Any]) -> dict[str, Any]:
        memory: dict[int, dict[str, Any]] = {}; local: dict[str, int] = {}
        next_sid = 100000; merge_count = 0; errors = []; max_states = int(self.data.config["model"]["max_deployed_novel_states"])

        def process(track_key: str, evaluator_category: int, phase: str) -> list[dict[str, Any]]:
            nonlocal next_sid, merge_count
            output = self.track_outputs(track_key); t = self.data.track_manifest[track_key]
            decisions = []
            for pos, (idx, token, predicted_ready_score) in enumerate(zip(output["indices"], output["tokens"], output["readiness"])):
                exact = bool(self.data.rows[int(idx)]["reliable"])
                ready = exact if self.exact_readiness else bool(float(predicted_ready_score) >= float(self.tau_ready))
                action = "DEFER"; sid = None; confidence = 1.0 - float(predicted_ready_score); evidence = "local_buffer_only"
                local_first = self.variant in {"b3", "no_merge"}
                if local_first and track_key in local:
                    sid = local[track_key]; action = "EXISTING_NOVEL"; confidence = 1.0; evidence = "old_local_first"
                elif not ready and track_key in local:
                    sid = local[track_key]; action = "EXISTING_NOVEL"; confidence = 1.0 - float(predicted_ready_score); evidence = "inherited_local_belief"
                elif not ready and self.variant != "b3":
                    pass
                else:
                    action, sid, confidence, extra = self.decode(token, memory, track_key, int(t["video_id"]))
                    evidence = "set_conditioned_decoder"
                    if action == "DEFER" and track_key in local:
                        action, sid, evidence = "EXISTING_NOVEL", local[track_key], "decoder_defer_inherited_local"
                    elif action == "NEW_NOVEL" and track_key in local:
                        action, sid, evidence = "EXISTING_NOVEL", local[track_key], "decoder_new_retained_local"
                    elif action == "NEW_NOVEL":
                        if len(memory) >= max_states:
                            action, sid, evidence = "DEFER", None, "memory_bound_defer"
                        else:
                            sid = next_sid; next_sid += 1
                            memory[sid] = {
                                "anchors": [], "anchor_scores": [], "birth_tracklet": track_key,
                                "birth_video": int(t["video_id"]),
                                "eval_category_not_model_input": evaluator_category,
                                "birth_row_key": self.data.rows[int(idx)]["row_key"], "aliased_to": None,
                            }
                            local[track_key] = sid; evidence = "novel_birth"
                    elif action == "EXISTING_NOVEL":
                        if sid not in memory:
                            errors.append(f"existing_before_birth:{sid}")
                            action, sid = "DEFER", None
                        else:
                            previous = local.get(track_key)
                            if previous is not None and previous != sid:
                                memory[previous]["aliased_to"] = sid; merge_count += 1
                            local[track_key] = int(sid)
                # B3 deliberately removes DEFER and therefore commits the current
                # observation immediately, including low-quality observations.
                # Other variants retain the registered reliable-only global update.
                if (ready or self.variant == "b3") and action in {"NEW_NOVEL", "EXISTING_NOVEL"} and sid in memory:
                    memory[int(sid)]["anchors"].append(self.data.row_input[int(idx)].astype(np.float32))
                    memory[int(sid)]["anchor_scores"].append(float(predicted_ready_score))
                    if len(memory[int(sid)]["anchors"]) > 8:
                        keep = np.argsort(-np.asarray(memory[int(sid)]["anchor_scores"]))[:8]
                        memory[int(sid)]["anchors"] = [memory[int(sid)]["anchors"][int(k)] for k in keep]
                        memory[int(sid)]["anchor_scores"] = [memory[int(sid)]["anchor_scores"][int(k)] for k in keep]
                decisions.append({
                    "row_key": self.data.rows[int(idx)]["row_key"], "tracklet_position": pos,
                    "phase": phase, "action": action, "semantic_id": sid,
                    "readiness_score": float(predicted_ready_score), "predicted_ready": ready,
                    "confidence": confidence, "evidence": evidence,
                    "state_count_after": len(memory),
                })
            return decisions

        source_cat = int(event.get("category_gt_denominator_only", event.get("distractor_category_gt_denominator_only")))
        source_decisions = []
        for key in event["source_tracklet_keys"]:
            source_decisions.extend(process(key, source_cat, "source"))
        target_cat = int(event.get("category_gt_denominator_only", event.get("target_category_gt_denominator_only")))
        target = process(event["target_tracklet_key"], target_cat, "target")
        prefix = int(event["target_first_reliable_prefix_index_gt_only"])

        def correct_existing(d: dict[str, Any]) -> bool:
            if d["action"] != "EXISTING_NOVEL" or d["semantic_id"] is None:
                return False
            m = memory.get(int(d["semantic_id"]))
            return bool(m and m["eval_category_not_model_input"] == target_cat
                        and m["birth_video"] != int(event["target_video"])
                        and m["birth_tracklet"] != event["target_tracklet_key"])

        post = target[prefix:]
        first_offset = next((i for i, d in enumerate(post) if d["action"] != "DEFER"), None)
        first = post[first_offset] if first_offset is not None else None
        correct_offsets = [i for i, d in enumerate(post) if correct_existing(d)]
        premature = [d for d in target[:prefix] if d["action"] != "DEFER"]
        return {
            "event_key": event["event_key"], "kind": event["kind"], "fold": event["fold"],
            "target_category_gt_evaluator_only": target_cat,
            "source_decisions": source_decisions, "target_decisions": target,
            "first_commit_after_prefix": first, "first_commit_offset": first_offset,
            "first_commit_correct_existing": bool(first and correct_existing(first)),
            "post_prefix_correct_existing_rows": len(correct_offsets), "post_prefix_rows": len(post),
            "time_to_correct_commit": min(correct_offsets) if correct_offsets else None,
            "pre_prefix_rows": prefix, "pre_prefix_defer_rows": prefix - len(premature),
            "premature_commit": bool(premature), "unresolved_after_prefix": first is None,
            "state_count": len(memory), "merge_count": merge_count,
            "duplicate_target_births": sum(m["eval_category_not_model_input"] == target_cat
                                           and m["birth_video"] == int(event["target_video"])
                                           for m in memory.values()),
            "transition_errors": errors,
        }


def calibration_indices(data: FoldData) -> list[int]:
    track_keys = [k for k, t in data.track_manifest.items()
                  if int(t["label_category_gt_only"]) in data.cal_cats
                  and int(t["video_id"]) not in data.held_videos]
    indices = {int(i) for k in track_keys for i in data.track_manifest[k]["row_indices"]}
    indices.update(i for i, r in enumerate(data.rows)
                   if r["role17"] in {"known_calibration", "novel_calibration"}
                   and r["video_i"] not in data.held_videos)
    return sorted(indices)


@torch.no_grad()
def readiness_threshold(model: DSTM, data: FoldData, device: torch.device,
                        variant: str) -> tuple[float, dict[str, Any], Runtime]:
    runtime = Runtime(model, data, device, variant, tau_ready=.5)
    indices = set(calibration_indices(data)); scores, labels = [], []
    keys = sorted({f"v{data.rows[i]['video_i']}:p{data.rows[i]['track_i']}" for i in indices})
    for key in keys:
        if key not in data.track_manifest:
            continue
        out = runtime.track_outputs(key)
        for idx, score in zip(out["indices"], out["readiness"]):
            if int(idx) in indices:
                scores.append(float(score)); labels.append(bool(data.rows[int(idx)]["reliable"]))
    s = np.asarray(scores, np.float32); y = np.asarray(labels, bool)
    tau, curve = threshold_curve(s, y, "f1")
    if variant == "repair_r1":
        feasible = [x for x in curve if x["recall"] >= .75]
        if feasible:
            feasible.sort(key=lambda x: (-x["precision"], -x["f1"], -x["threshold"]))
            tau = feasible[0]["threshold"]
            selected = feasible[0]
        else:
            selected = max(curve, key=lambda x: (x["recall"], x["precision"], -x["threshold"]))
            tau = selected["threshold"]
    else:
        selected = curve[0]
    runtime.tau_ready = tau
    return tau, {
        "rows": len(y), "positives": int(y.sum()), "threshold": tau,
        "auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
        "selected": selected,
        "selection_rule": ("maximum precision subject to recall >= 0.75" if variant == "repair_r1" else "maximum F1"),
    }, runtime


def known_action_metrics(runtime: Runtime, role: str) -> dict[str, Any]:
    data = runtime.data
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(data.rows):
        if r["role17"] == role and r["gt_role_common"] == "supported_known":
            if role == "known_calibration" and r["video_i"] in data.held_videos:
                continue
            groups[f"v{r['video_i']}:p{r['track_i']}"].append(i)
    correct = []; by: dict[int, list[int]] = defaultdict(list)
    for key, target_indices in sorted(groups.items()):
        if key not in data.track_manifest:
            continue
        out = runtime.track_outputs(key); position = {int(i): p for p, i in enumerate(out["indices"])}
        local_known = None
        for i in sorted(target_indices, key=lambda x: data.rows[x]["event_i"]):
            pos = position[i]; score = float(out["readiness"][pos])
            ready = bool(data.rows[i]["reliable"]) if runtime.exact_readiness else score >= float(runtime.tau_ready)
            if ready or runtime.variant == "b3":
                action, sid, _, _ = runtime.decode(out["tokens"][pos], {}, key, data.rows[i]["video_i"])
                if action == "KNOWN":
                    local_known = sid
            pred = local_known if local_known is not None else -1
            ok = int(pred == data.rows[i]["cat_i"]); correct.append(ok); by[data.rows[i]["cat_i"]].append(ok)
    return {
        "rows": len(correct), "micro_accuracy": float(np.mean(correct)) if correct else 0.0,
        "category_macro_accuracy": float(np.mean([np.mean(v) for v in by.values()])) if by else 0.0,
        "categories": len(by),
        "by_category": {str(c): {"correct": sum(v), "rows": len(v), "accuracy": float(np.mean(v))}
                        for c, v in sorted(by.items())},
    }


def evaluate_calibration(model: DSTM, data: FoldData, device: torch.device,
                         variant: str, include_records: bool = False) -> dict[str, Any]:
    tau, reliability, runtime = readiness_threshold(model, data, device, variant)
    events = build_category_events(data, sorted(data.cal_cats))
    records = [runtime.simulate_event(e) for e in events]
    metrics = event_metrics(records)
    known = known_action_metrics(runtime, "known_calibration")
    coverage = 1.0 - metrics["unresolved_event_rate"]
    latency = metrics["mean_time_to_correct_commit"]
    latency_norm = 1.0 if latency is None else min(1.0, float(latency) / 32.0)
    valid = all(not r["transition_errors"] for r in records)
    composite = (2.0 * metrics["commit_ct"]["recall"] + .5 * metrics["existing_precision"]
                 - .5 * metrics["negative_false_merge_rate"] + .5 * known["category_macro_accuracy"]
                 + .2 * coverage - .05 * latency_norm + .25 * float(valid))
    result = {
        "fold": data.fold_id, "variant": variant, "tau_ready": tau,
        "reliability": reliability, "metrics": metrics, "known": known,
        "transition_valid": valid, "composite": composite,
        "events": len(events), "selected_from_calibration_only": True,
    }
    if include_records:
        result["event_records"] = records
    return result


def held_events(data: FoldData) -> list[dict[str, Any]]:
    pos = [x for x in load_jsonl(OUT / "episodes/identifiable_positive_events.jsonl") if int(x["fold"]) == data.fold_id]
    neg = [x for x in load_jsonl(OUT / "episodes/identifiable_negative_events.jsonl") if int(x["fold"]) == data.fold_id]
    return sorted(pos + neg, key=lambda x: x["event_key"])


def evaluate_held(model: DSTM, data: FoldData, device: torch.device,
                  variant: str, exact_readiness: bool = False) -> dict[str, Any]:
    tau, cal_reliability, _ = readiness_threshold(model, data, device, variant)
    runtime = Runtime(model, data, device, variant, tau_ready=tau, exact_readiness=exact_readiness)
    events = held_events(data)
    records = [runtime.simulate_event(e) for e in events]
    metrics = event_metrics(records)
    known = known_action_metrics(runtime, "known_audit")
    seen, scores, labels = set(), [], []
    for e in events:
        if e["kind"] != "positive_existing":
            continue
        key = e["target_tracklet_key"]; out = runtime.track_outputs(key)
        for idx, score in zip(out["indices"], out["readiness"]):
            if int(idx) in seen:
                continue
            seen.add(int(idx)); scores.append(float(score)); labels.append(bool(data.rows[int(idx)]["reliable"]))
    s = np.asarray(scores, np.float32); y = np.asarray(labels, bool); pred = s >= tau
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    reliability = {
        "rows": len(y), "positive_rows": int(y.sum()), "tau_ready": tau,
        "auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
    }
    return {
        "fold": data.fold_id, "variant": variant, "tau_ready": tau,
        "calibration_reliability": cal_reliability, "metrics": metrics,
        "known": known, "reliability": reliability, "event_records": records,
        "exact_readiness_control": exact_readiness,
        "transition_valid": all(not r["transition_errors"] for r in records),
    }
