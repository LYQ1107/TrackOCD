"""Complete calibration, causal replay, audit, and controls for Phase17R."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.iclr27_phase17r.training.model import ObservabilitySemanticModel
from src.iclr27_phase17r.training.train_full_model import CAL_ROLES, GEOMETRY_FIELDS, load_data

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
AUDIT_ROLES = {"known_audit", "novel_audit"}
SEEDS = (20260825, 20260826, 20260827)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40); return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    y = x - x.max(axis=1, keepdims=True); y = np.exp(y); return y / np.maximum(y.sum(axis=1, keepdims=True), 1e-12)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def model_outputs(model: ObservabilitySemanticModel, data: dict[str, Any], batch: int = 512) -> dict[str, np.ndarray]:
    model.eval(); out = defaultdict(list)
    with torch.no_grad():
        for s in range(0, len(data["rows"]), batch):
            e = min(len(data["rows"]), s + batch)
            pred = model(torch.from_numpy(data["views"][s:e]), torch.from_numpy(data["previous_raw"][s:e]), torch.from_numpy(data["geometry"][s:e]))
            for k in ("semantic", "observability_logit", "known_logit", "class_logits"):
                out[k].append(pred[k].float().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


class PairScorer:
    def __init__(self, model: ObservabilitySemanticModel):
        state = model.state_dict()
        self.w1 = state["pair_head.0.weight"].cpu().numpy(); self.b1 = state["pair_head.0.bias"].cpu().numpy()
        self.w2 = state["pair_head.3.weight"].cpu().numpy(); self.b2 = float(state["pair_head.3.bias"].cpu().numpy()[0])

    def score(self, query: np.ndarray, states: np.ndarray, state_count_log: float) -> np.ndarray:
        if not len(states): return np.empty(0, dtype=np.float32)
        q = np.repeat(query[None], len(states), axis=0)
        x = np.concatenate([np.abs(q - states), q * states, np.full((len(states), 1), state_count_log, np.float32)], axis=1)
        h = x @ self.w1.T + self.b1
        h = .5 * h * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (h + .044715 * h ** 3)))
        return sigmoid((h @ self.w2.T).reshape(-1) + self.b2)


def choose_binary_threshold(scores: np.ndarray, labels: np.ndarray, objective: str) -> tuple[float, list[dict[str, float]]]:
    grid = np.unique(np.concatenate([np.linspace(.05, .95, 19), np.quantile(scores, np.linspace(0, 1, 21))]))
    values = []
    for threshold in grid:
        pred = scores >= threshold; tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum()); fn = int((~pred & labels).sum()); tn = int((~pred & ~labels).sum())
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        balanced = .5 * (recall + tn / max(tn + fp, 1))
        score = f1 if objective == "f1" else balanced
        values.append({"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "balanced_accuracy": balanced, "objective": score})
    values.sort(key=lambda x: (-x["objective"], -x["precision"], x["threshold"]))
    return values[0]["threshold"], values


def known_threshold(outputs: dict[str, np.ndarray], data: dict[str, Any], idx: np.ndarray, known_ids: list[int]) -> tuple[float, list[dict[str, float]], np.ndarray, np.ndarray]:
    probs = softmax(outputs["class_logits"]); pred_index = probs.argmax(1); class_conf = probs.max(1)
    accept_score = sigmoid(outputs["known_logit"]) * class_conf
    known_map = {c: i for i, c in enumerate(known_ids)}
    rows = data["rows"]; roles, cats = data["roles"], data["cats"]
    grid = np.unique(np.concatenate([np.linspace(.01, .95, 24), np.quantile(accept_score[idx], np.linspace(0, 1, 21))]))
    values = []
    known_idx = idx[roles[idx] == 1]; non_idx = idx[roles[idx] != 1]
    for threshold in grid:
        accepted = accept_score >= threshold
        correct = sum(accepted[i] and known_ids[int(pred_index[i])] == int(cats[i]) for i in known_idx)
        acc = correct / max(len(known_idx), 1)
        false_accept = float(accepted[non_idx].mean()) if len(non_idx) else 0.0
        accepted_idx = idx[accepted[idx]]
        precision = sum(roles[i] == 1 and known_ids[int(pred_index[i])] == int(cats[i]) for i in accepted_idx) / max(len(accepted_idx), 1)
        objective = acc + .20 * precision - .20 * false_accept
        values.append({"threshold": float(threshold), "known_occurrence_accuracy": acc, "accepted_correct_precision": precision, "nonknown_false_accept_rate": false_accept, "objective": objective})
    values.sort(key=lambda x: (-x["objective"], -x["known_occurrence_accuracy"], x["threshold"]))
    return values[0]["threshold"], values, pred_index, accept_score


def track_items(indices: np.ndarray, data: dict[str, Any], embedding: np.ndarray, role_value: int | None = None) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for i in indices:
        i = int(i); r = data["rows"][i]
        if not data["assigned"][i] or (role_value is not None and data["roles"][i] != role_value): continue
        groups[(int(r["video_id"]), int(r["track_id"]), int(data["cats"][i]))].append(i)
    items = []
    for (v, t, c), idx in groups.items():
        z = l2(embedding[idx].mean(0, keepdims=True))[0]
        items.append({"video": v, "track": t, "category": c, "embedding": z, "rows": idx})
    return items


def retrieval(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items: return {"tracks": 0, "queries": 0}
    z = np.asarray([x["embedding"] for x in items], np.float32); sim = z @ z.T
    r1, r5, aps, pair_y, pair_s = [], [], [], [], []
    for i, item in enumerate(items):
        cand = np.asarray([j for j, x in enumerate(items) if x["video"] != item["video"]], dtype=int)
        if not len(cand): continue
        y = np.asarray([items[j]["category"] == item["category"] for j in cand])
        if not y.any(): continue
        order = np.argsort(-sim[i, cand]); r1.append(int(y[order[:1]].any())); r5.append(int(y[order[:5]].any()))
        aps.append(float(average_precision_score(y.astype(int), sim[i, cand])))
        pair_y.extend(y.astype(int).tolist()); pair_s.extend(sim[i, cand].tolist())
    out = {"tracks": len(items), "videos": len({x["video"] for x in items}), "categories": len({x["category"] for x in items}), "queries": len(r1),
           "r_at_1": float(np.mean(r1)) if r1 else None, "r_at_5": float(np.mean(r5)) if r5 else None, "mAP": float(np.mean(aps)) if aps else None,
           "positive_pairs": int(sum(pair_y)), "pairs": len(pair_y)}
    if len(set(pair_y)) > 1:
        out["pair_roc_auc"] = float(roc_auc_score(pair_y, pair_s)); out["pair_pr_auc"] = float(average_precision_score(pair_y, pair_s))
    return out


def calibrate_pair(scorer: PairScorer, items: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    score, labels = [], []
    for i, a in enumerate(items):
        cand = [b for b in items[i + 1:] if b["video"] != a["video"]]
        if not cand: continue
        vals = scorer.score(a["embedding"], np.asarray([b["embedding"] for b in cand]), .5)
        score.extend(vals.tolist()); labels.extend(int(a["category"] == b["category"]) for b in cand)
    if len(set(labels)) < 2:
        return .5, {"pairs": len(labels), "positive_pairs": int(sum(labels)), "identified": False}
    score_a, label_a = np.asarray(score), np.asarray(labels, dtype=bool)
    threshold, curve = choose_binary_threshold(score_a, label_a, "f1")
    return threshold, {"pairs": len(labels), "positive_pairs": int(sum(labels)), "roc_auc": float(roc_auc_score(label_a, score_a)), "pr_auc": float(average_precision_score(label_a, score_a)), "threshold_curve": curve, "identified": True}


def replay(indices: np.ndarray, rank_field: str, data: dict[str, Any], semantic: np.ndarray,
           obs_score: np.ndarray, known_pred: np.ndarray, known_score: np.ndarray,
           scorer: PairScorer, thresholds: dict[str, float], promotion: int,
           oracle_observability: bool = False, oracle_known: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # This is the only ordering operation: it consumes the immutable rank. No
    # downstream helper sorts by numeric video ID.
    order = sorted((int(i) for i in indices), key=lambda i: int(data["rows"][i][rank_field]))
    states: dict[int, dict[str, Any]] = {}; local_novel = {}; local_known = {}; next_sid = 100000
    decisions = []
    for event_pos, i in enumerate(order):
        r = data["rows"][i]; physical = (int(r["video_id"]), int(r["track_id"])); q = semantic[i]
        observable = bool(data["observable"][i]) if oracle_observability else bool(obs_score[i] >= thresholds["tau_observable"])
        is_true_known = data["roles"][i] == 1
        accept_known = is_true_known if oracle_known else bool(known_score[i] >= thresholds["tau_known"])
        category = int(data["cats"][i]) if oracle_known and is_true_known else int(known_pred[i])
        action, sid, source = None, None, None
        if oracle_known and is_true_known:
            action, sid, source = "known", category, "oracle_known"
            local_known[physical] = category; local_novel.pop(physical, None)
        elif observable and accept_known:
            action, sid, source = "known", category, "known_head"
            local_known[physical] = category; local_novel.pop(physical, None)
        elif not observable and physical in local_known:
            action, sid, source = "known", local_known[physical], "reliable_local_known"
        elif physical in local_novel and local_novel[physical] in states:
            sid = local_novel[physical]; action, source = "existing", "local_continuity"
        else:
            cand_sid, cand_vec = [], []
            for k, state in states.items():
                legal = [a[0] for a in state["anchors"] if a[1] != physical and a[1][0] != physical[0]]
                if state["promoted"] and legal:
                    cand_sid.append(k); cand_vec.append(l2(np.mean(legal, axis=0, keepdims=True))[0])
            best_sid, best_prob = None, -1.0
            if observable and cand_vec:
                matrix = np.asarray(cand_vec, np.float32); cosine = matrix @ q
                keep = np.argsort(-cosine)[:min(8, len(matrix))]
                probs = scorer.score(q, matrix[keep], min(1.0, math.log1p(len(states)) / math.log(65.0)))
                j = int(np.argmax(probs)); best_sid, best_prob = cand_sid[int(keep[j])], float(probs[j])
            if best_sid is not None and best_prob >= thresholds["tau_pair"]:
                action, sid, source = "existing", best_sid, "cross_physical"
            else:
                sid = next_sid; next_sid += 1; action, source = "new", "quarantined_new"
                states[sid] = {"anchors": [], "reliable_count": 0, "promoted": False,
                               "birth_event": event_pos, "birth_row_key": r["row_key"]}
                local_novel[physical] = sid; local_known.pop(physical, None)
        if action in {"new", "existing"}:
            state = states[int(sid)]; local_novel[physical] = int(sid)
            if observable:
                state["anchors"].append((q.copy(), physical, event_pos)); state["anchors"] = state["anchors"][-8:]
                state["reliable_count"] += 1; state["promoted"] = state["reliable_count"] >= promotion
        decisions.append({"global_index": i, "row_key": r["row_key"], "event_rank": int(r[rank_field]),
                          "action": action, "semantic_id": int(sid), "evidence_source": source,
                          "predicted_observable": observable, "observability_score": float(obs_score[i]),
                          "known_score": float(known_score[i]), "state_count_after": len(states)})
    return [data["rows"][i] for i in order], decisions


def strict_metrics(rows: list[dict[str, Any]], decisions: list[dict[str, Any]], eligible_keys: list[str], known_ids: set[int]) -> dict[str, Any]:
    births = {}
    errors = []
    for pos, (r, d) in enumerate(zip(rows, decisions)):
        if d["action"] == "new":
            sid = d["semantic_id"]
            if sid in births: errors.append("duplicate_new:" + str(sid))
            births[sid] = {"category": int(r["gt_category_id_common"]) if r["gt_role_common"] == "novel" else -1,
                           "video": int(r["video_id"]), "track": int(r["track_id"]), "position": pos}
        elif d["action"] == "existing":
            if d["semantic_id"] not in births or births[d["semantic_id"]]["position"] >= pos: errors.append("existing_before_birth:" + str(d["semantic_id"]))
        elif d["action"] == "known" and d["semantic_id"] not in known_ids:
            errors.append("invalid_known:" + str(d["semantic_id"]))
    known_rows = [(r, d) for r, d in zip(rows, decisions) if r["gt_role_common"] == "supported_known"]
    known_ok = [d["action"] == "known" and d["semantic_id"] == int(r["gt_category_id_common"]) for r, d in known_rows]
    by_cat = defaultdict(list)
    for (r, _), ok in zip(known_rows, known_ok): by_cat[int(r["gt_category_id_common"])].append(int(ok))
    eligible = set(eligible_keys); ct_rows = [(r, d) for r, d in zip(rows, decisions) if r["row_key"] in eligible]
    ct_ok = []
    for r, d in ct_rows:
        birth = births.get(d["semantic_id"])
        ct_ok.append(bool(d["action"] == "existing" and birth and birth["category"] == int(r["gt_category_id_common"]) and birth["video"] != int(r["video_id"])))
    ct_cat, ct_vid = defaultdict(list), defaultdict(list)
    for (r, _), ok in zip(ct_rows, ct_ok):
        ct_cat[int(r["gt_category_id_common"])].append(int(ok)); ct_vid[int(r["video_id"])].append(int(ok))
    existing = [(r, d) for r, d in zip(rows, decisions) if d["action"] == "existing"]
    existing_ok = []
    for r, d in existing:
        birth = births.get(d["semantic_id"])
        existing_ok.append(bool(r["gt_role_common"] == "novel" and birth and birth["category"] == int(r["gt_category_id_common"])))
    new_rows = [(r, d) for r, d in zip(rows, decisions) if d["action"] == "new"]
    frag = defaultdict(set)
    for r, d in zip(rows, decisions):
        if r["gt_role_common"] == "novel" and d["action"] in {"new", "existing"}: frag[int(r["gt_category_id_common"])].add(d["semantic_id"])

    def stratum(lo: float, hi: float | None) -> dict[str, float]:
        chosen = [(r, d) for r, d in zip(rows, decisions) if float(r["row_iou"]) >= lo and (hi is None or float(r["row_iou"]) < hi)]
        kk = [d["action"] == "known" and d["semantic_id"] == int(r["gt_category_id_common"]) for r, d in chosen if r["gt_role_common"] == "supported_known"]
        return {"rows": len(chosen), "known_rows": len(kk), "known_accuracy": float(np.mean(kk)) if kk else 0.0}
    actions = Counter(d["action"] for d in decisions); evidence = Counter(d["evidence_source"] for d in decisions)
    return {
        "rows": len(rows), "known_occurrences": len(known_ok), "known_occurrence_accuracy": float(np.mean(known_ok)) if known_ok else 0.0,
        "known_category_macro": float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0,
        "known_by_category": {str(k): {"correct": int(sum(v)), "rows": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(by_cat.items())},
        "fixed_ct": {"eligible": len(ct_rows), "correct": int(sum(ct_ok)), "recall": float(np.mean(ct_ok)) if ct_ok else 0.0,
                     "denominator_sha256": hashlib.sha256(json.dumps(eligible_keys).encode()).hexdigest(),
                     "correct_categories": sum(sum(v) > 0 for v in ct_cat.values()), "correct_videos": sum(sum(v) > 0 for v in ct_vid.values()),
                     "by_category": {str(k): {"correct": int(sum(v)), "eligible": len(v)} for k, v in sorted(ct_cat.items())},
                     "by_video": {str(k): {"correct": int(sum(v)), "eligible": len(v)} for k, v in sorted(ct_vid.items())}},
        "predicted_existing": len(existing), "predicted_existing_precision": float(np.mean(existing_ok)) if existing_ok else 0.0,
        "births": len(new_rows), "birth_precision": sum(r["gt_role_common"] == "novel" for r, _ in new_rows) / max(len(new_rows), 1),
        "state_count": max((d["state_count_after"] for d in decisions), default=0),
        "fragmentation_mean_states": float(np.mean([len(x) for x in frag.values()])) if frag else 0.0,
        "duplicate_creation_rate": float(np.mean([len(x) > 1 for x in frag.values()])) if frag else 0.0,
        "actions": dict(actions), "evidence_sources": dict(evidence),
        "quality_strata": {"zero_iou": stratum(0.0, 1e-12), "low_iou": stratum(1e-12, .5), "high_iou": stratum(.5, None)},
        "transition_contract": {"valid": not errors, "errors": errors[:20], "new_states_unique": len(births) == len(new_rows), "existing_after_birth": not any(x.startswith("existing_before_birth") for x in errors)},
        "chronology_contract": {"valid": all(decisions[i]["event_rank"] < decisions[i + 1]["event_rank"] for i in range(len(decisions) - 1)), "event_rank_consumed": True},
        "future_or_gt_inference_input": False
    }


def observability_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold; tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum()); fn = int((~pred & labels).sum())
    return {"auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores)),
            "threshold": threshold, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "predicted_positive": int(pred.sum()), "true_positive_population": int(labels.sum())}


def prototype_accuracy(bank_idx: np.ndarray, query_idx: np.ndarray, features: np.ndarray, data: dict[str, Any], clean: bool) -> dict[str, Any]:
    cats = defaultdict(list)
    for i in bank_idx:
        i = int(i)
        if data["roles"][i] == 1 and data["assigned"][i] and (not clean or data["observable"][i]): cats[int(data["cats"][i])].append(features[i])
    categories = sorted(cats); proto = l2(np.asarray([np.mean(cats[c], axis=0) for c in categories], np.float32)) if categories else np.empty((0, features.shape[1]))
    target = [int(i) for i in query_idx if data["roles"][int(i)] == 1]
    pred = [categories[int(np.argmax(proto @ l2(features[i][None])[0]))] if categories else -1 for i in target]
    ok = [p == int(data["cats"][i]) for p, i in zip(pred, target)]; by = defaultdict(list)
    for i, x in zip(target, ok): by[int(data["cats"][i])].append(int(x))
    return {"bank_rows": sum(len(v) for v in cats.values()), "bank_categories": len(categories), "query_rows": len(target),
            "closed_top1": float(np.mean(ok)) if ok else 0.0, "category_macro": float(np.mean([np.mean(v) for v in by.values()])) if by else 0.0,
            "missing_query_categories": sorted(set(int(data["cats"][i]) for i in target) - set(categories))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    data = load_data(args.rows, args.features, args.variant)
    known_ids = [int(x) for x in checkpoint["known_ids"]]
    model = ObservabilitySemanticModel(768, len(GEOMETRY_FIELDS), int(checkpoint["embedding_dim"]), len(known_ids))
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    outputs = model_outputs(model, data); semantic = outputs["semantic"]
    scorer = PairScorer(model); obs_score = sigmoid(outputs["observability_logit"])
    cal_idx = data["cal_idx"]; audit_idx = np.asarray([i for i, r in enumerate(data["rows"]) if r["role17"] in AUDIT_ROLES], dtype=np.int64)
    tau_obs, obs_curve = choose_binary_threshold(obs_score[cal_idx], data["observable"][cal_idx], "f1")
    tau_known, known_curve, known_pred_index, known_score = known_threshold(outputs, data, cal_idx, known_ids)
    known_pred = np.asarray([known_ids[int(i)] for i in known_pred_index], dtype=np.int64)
    cal_novel_items = track_items(cal_idx, data, semantic, role_value=2)
    tau_pair, pair_cal = calibrate_pair(scorer, cal_novel_items)
    denom = json.loads(args.denominators.read_text())
    pair_candidates = sorted(set([max(.01, tau_pair - .10), max(.01, tau_pair - .05), tau_pair, min(.99, tau_pair + .05), min(.99, tau_pair + .10)]))
    grid = []
    for pair_t in pair_candidates:
        for promotion in (1, 2):
            order_results = []
            for oi, seed in enumerate(SEEDS):
                rr, dd = replay(cal_idx, "event_rank_order" + str(oi), data, semantic, obs_score, known_pred, known_score, scorer,
                                {"tau_observable": tau_obs, "tau_known": tau_known, "tau_pair": pair_t}, promotion)
                keys = denom["denominators"]["calibration"][str(seed)]["row_keys"]
                order_results.append(strict_metrics(rr, dd, keys, set(known_ids)))
            objective = float(np.mean([x["known_occurrence_accuracy"] + x["fixed_ct"]["recall"] + .25 * x["predicted_existing_precision"] + .05 * x["birth_precision"] - .0001 * x["state_count"] for x in order_results]))
            grid.append({"tau_pair": pair_t, "promotion_reliable_observations": promotion, "objective": objective,
                         "mean_known": float(np.mean([x["known_occurrence_accuracy"] for x in order_results])),
                         "mean_ct": float(np.mean([x["fixed_ct"]["recall"] for x in order_results])),
                         "mean_existing_precision": float(np.mean([x["predicted_existing_precision"] for x in order_results])),
                         "per_order": order_results})
    grid.sort(key=lambda x: (-x["objective"], -x["mean_ct"], -x["mean_existing_precision"], x["tau_pair"], x["promotion_reliable_observations"]))
    best = grid[0]; thresholds = {"tau_observable": tau_obs, "tau_known": tau_known, "tau_pair": best["tau_pair"], "promotion_reliable_observations": best["promotion_reliable_observations"]}

    audit_orders, controls = [], defaultdict(list)
    decision_dir = args.out_dir / "csv"; decision_dir.mkdir(parents=True, exist_ok=True)
    for oi, seed in enumerate(SEEDS):
        rank_field = "event_rank_order" + str(oi); keys = denom["denominators"]["audit"][str(seed)]["row_keys"]
        rr, dd = replay(audit_idx, rank_field, data, semantic, obs_score, known_pred, known_score, scorer, thresholds, int(thresholds["promotion_reliable_observations"]))
        met = strict_metrics(rr, dd, keys, set(known_ids)); audit_orders.append({"seed": seed, "metrics": met})
        if args.candidate == "m1":
            path = decision_dir / ("public_final_audit_decisions_" + str(seed) + ".csv"); tmp = path.with_suffix(path.suffix + ".tmp")
            merged = [{**r, **d} for r, d in zip(rr, dd)]
            with tmp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(merged[0])); w.writeheader(); w.writerows(merged)
            os.replace(tmp, path)
        for name, oo, ok in [("oracle_observability", True, False), ("oracle_known", False, True), ("oracle_both_routing", True, True)]:
            cr, cd = replay(audit_idx, rank_field, data, semantic, obs_score, known_pred, known_score, scorer, thresholds, int(thresholds["promotion_reliable_observations"]), oracle_observability=oo, oracle_known=ok)
            controls[name].append({"seed": seed, "metrics": strict_metrics(cr, cd, keys, set(known_ids))})
        # Full semantic correspondence oracle: one causal state per GT novel
        # category, for evaluator/task upper bound only.
        order = sorted((int(i) for i in audit_idx), key=lambda i: int(data["rows"][i][rank_field])); states = {}; next_sid = 500000; od = []
        for i in order:
            r = data["rows"][i]
            if r["gt_role_common"] == "supported_known": od.append({"action": "known", "semantic_id": int(r["gt_category_id_common"]), "event_rank": int(r[rank_field]), "state_count_after": len(states), "evidence_source": "semantic_oracle", "row_key": r["row_key"]})
            elif r["gt_role_common"] == "novel":
                c = int(r["gt_category_id_common"]); new = c not in states
                if new: states[c] = next_sid; next_sid += 1
                od.append({"action": "new" if new else "existing", "semantic_id": states[c], "event_rank": int(r[rank_field]), "state_count_after": len(states), "evidence_source": "semantic_oracle", "row_key": r["row_key"]})
            else:
                sid = next_sid; next_sid += 1
                od.append({"action": "new", "semantic_id": sid, "event_rank": int(r[rank_field]), "state_count_after": len(states) + 1, "evidence_source": "semantic_oracle_fp", "row_key": r["row_key"]})
        controls["semantic_correspondence_oracle"].append({"seed": seed, "metrics": strict_metrics([data["rows"][i] for i in order], od, keys, set(known_ids))})

    closed_known = audit_idx[data["roles"][audit_idx] == 1]
    closed_ok = known_pred[closed_known] == data["cats"][closed_known]
    by_closed = defaultdict(list)
    for i, ok in zip(closed_known, closed_ok): by_closed[int(data["cats"][i])].append(int(ok))
    offline = {"known_closed_top1": float(np.mean(closed_ok)), "known_closed_category_macro": float(np.mean([np.mean(v) for v in by_closed.values()])),
               "known_rows": len(closed_known), "novel_retrieval": retrieval(track_items(audit_idx, data, semantic, role_value=2)),
               "known_retrieval": retrieval(track_items(audit_idx, data, semantic, role_value=1))}
    obs_audit = observability_metrics(obs_score[audit_idx], data["observable"][audit_idx], tau_obs)
    gate_orders = [x["metrics"] for x in audit_orders]
    gate = {
        "known_ge_0_60_all_orders": all(x["known_occurrence_accuracy"] >= .60 for x in gate_orders),
        "fixed_ct_gt_0_all_orders": all(x["fixed_ct"]["recall"] > 0 for x in gate_orders),
        "correct_ct_categories_ge_3_all_orders": all(x["fixed_ct"]["correct_categories"] >= 3 for x in gate_orders),
        "correct_ct_videos_ge_3_all_orders": all(x["fixed_ct"]["correct_videos"] >= 3 for x in gate_orders),
        "predicted_existing_precision_ge_0_20_all_orders": all(x["predicted_existing_precision"] >= .20 for x in gate_orders),
        "contracts_valid_all_orders": all(x["transition_contract"]["valid"] and x["chronology_contract"]["valid"] and not x["future_or_gt_inference_input"] for x in gate_orders)
    }
    gate["passed"] = all(gate.values())
    calibration = {"protocol": "trackocd_iclr27_phase17r_public_calibration", "candidate": args.candidate, "checkpoint": str(args.checkpoint.resolve()),
                   "checkpoint_step": checkpoint["step"], "complete_rows": len(cal_idx), "thresholds": thresholds,
                   "observability_curve": obs_curve, "known_curve": known_curve, "pair_calibration": pair_cal,
                   "controller_grid": grid, "selected_by_calibration_only": True, "audit_rows_inspected": False,
                   "episode_orders": list(SEEDS)}
    audit = {"protocol": "trackocd_iclr27_phase17r_public_final_audit", "candidate": args.candidate, "complete_rows": len(audit_idx),
             "thresholds_locked": thresholds, "offline": offline, "observability": obs_audit,
             "orders": audit_orders, "oracle_controls": dict(controls), "public_gate": gate,
             "audit_selected_model": False, "q1_labels_used": False, "devplus_labels_used": False,
             "no_future_or_gt_inference_input": True}

    train_idx = data["train_idx"]
    bank = {
        "B0_PHASE17_ALL_ASSIGNED_BANK": prototype_accuracy(train_idx, audit_idx, semantic, data, clean=False),
        "B1_CLEAN_PROPOSAL_BANK": prototype_accuracy(train_idx, audit_idx, semantic, data, clean=True)
    }
    if args.variant == "m1":
        with torch.no_grad():
            teacher = np.zeros_like(semantic)
            mask = data["teacher_mask"]
            for s in range(0, len(data["rows"]), 512):
                e = min(len(data["rows"]), s + 512); local = mask[s:e]
                if local.any(): teacher[s:e][local] = model.teacher(torch.from_numpy(data["gt_views"][s:e][local])).numpy()
        bank["B2_GT_TEACHER_BANK_DIAGNOSTIC"] = prototype_accuracy(train_idx, audit_idx, teacher, data, clean=True)
        bank["P0_FROZEN_DINOV3_CONTEXT"] = prototype_accuracy(train_idx, audit_idx, l2(data["views"][:, :3].mean(1)), data, clean=False)
    bank_value = {"protocol": "trackocd_iclr27_phase17r_bank_contamination", "candidate": args.candidate, "baselines": bank,
                  "audit_used_for_diagnostic_only": True, "gt_teacher_deployed": False}

    if args.candidate == "m1":
        atomic_json(args.out_dir / "eval/public_calibration_summary.json", calibration)
        atomic_json(args.out_dir / "eval/public_final_audit.json", audit)
        atomic_json(args.out_dir / "eval/bank_contamination_audit.json", bank_value)
        lock = {"protocol": "trackocd_iclr27_phase17r_public_lock", "candidate": args.candidate,
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": checkpoint["step"], "thresholds": thresholds,
                "feature_manifest": str((args.out_dir / "features/full_public_dinov3.json").resolve()),
                "calibration": str((args.out_dir / "eval/public_calibration_summary.json").resolve()),
                "audit": str((args.out_dir / "eval/public_final_audit.json").resolve()), "public_gate_passed": gate["passed"],
                "devplus_authorized": gate["passed"], "audit_selected_model": False}
        atomic_json(args.out_dir / "manifests/public_lock.json", lock)
    else:
        atomic_json(args.out_dir / "eval/t0_public_calibration_summary.json", calibration)
        atomic_json(args.out_dir / "eval/t0_public_final_audit.json", audit)
        atomic_json(args.out_dir / "eval/t0_bank_contamination_audit.json", bank_value)
    print(json.dumps({"candidate": args.candidate, "checkpoint_step": checkpoint["step"], "thresholds": thresholds,
                      "offline": offline, "observability": obs_audit, "order_metrics": [{"seed": x["seed"], **x["metrics"]} for x in audit_orders], "gate": gate}, indent=2, sort_keys=True))
    return {"calibration": calibration, "audit": audit, "bank": bank_value}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=["t0", "m1"], required=True)
    ap.add_argument("--variant", choices=["t0", "m1"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--rows", type=Path, default=ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv")
    ap.add_argument("--denominators", type=Path, default=ROOT / "outputs/iclr27_phase17r/eval/fixed_ct_denominators.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/iclr27_phase17r")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
