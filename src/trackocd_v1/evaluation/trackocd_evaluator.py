#!/usr/bin/env python3
"""TrackOCD-v1.0 corrected evaluator.

Canonical prediction label space (per track):
  {"prediction_type": "known", "semantic_category_id": <official id>}
  {"prediction_type": "novel", "virtual_category_id": <int>}
  {"prediction_type": "unresolved"}

Rules:
- Known GT tracks are correct only if predicted as `known` with the exact
  semantic category id. known->novel, known->wrong-known, known->unresolved
  are all errors.
- Novel GT tracks are correct only if predicted as `novel`; the virtual id is
  matched to GT novel categories by Hungarian matching over the novel-only
  contingency matrix. novel->known and novel->unresolved are routing errors
  and never enter Hungarian.
- Known semantic ids never participate in the novel Hungarian matching.

Both Route-aware Novel ACC (denominator = all novel tracks) and Conditional
Novel ACC (denominator = correctly routed novel tracks) are reported, along
with Novel Routing Recall so conditional numbers cannot hide routing failure.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def hungarian_match(count_matrix, row_ids, col_ids):
    """Max-weight bipartite matching. count_matrix rows=pred, cols=gt."""
    rows, cols = linear_sum_assignment(-count_matrix)
    return {row_ids[r]: col_ids[c] for r, c in zip(rows, cols)}


class TrackOCDEvaluator:
    def __init__(self, gt_rows):
        """gt_rows: list of {sample_id, ground_truth_category_id,
        protocol_role} (protocol_role in supported_known/zero_shot_known/
        novel/distractor)."""
        self.gt = {}
        for r in gt_rows:
            self.gt[r["sample_id"]] = r

    def evaluate(self, preds, subset_ids=None, metadata=None):
        pred_map = {}
        for p in preds:
            pred_map[p["sample_id"]] = p

        rows = []
        for sid, g in self.gt.items():
            role = g["protocol_role"]
            if role == "distractor":
                continue
            if subset_ids is not None and sid not in subset_ids:
                continue
            rows.append((sid, g, pred_map.get(sid)))

        known_mask = np.array(
            [g["protocol_role"] in ("supported_known", "zero_shot_known") for _, g, _ in rows]
        )
        novel_mask = np.array([g["protocol_role"] == "novel" for _, g, _ in rows])
        gt_cat = np.array([g["ground_truth_category_id"] for _, g, _ in rows])
        ptype = np.array([(p or {}).get("prediction_type", "unresolved") for _, _, p in rows])
        sem_id = np.array(
            [(p or {}).get("semantic_category_id") for _, _, p in rows]
        )
        virt_id = np.array(
            [(p or {}).get("virtual_category_id") for _, _, p in rows]
        )
        stream_order = np.array(
            [(p or {}).get("stream_order", -1) for _, _, p in rows]
        )

        n_known = int(known_mask.sum())
        n_novel = int(novel_mask.sum())

        # ---- Known metrics ----
        known_correct = known_mask & (ptype == "known") & (
            np.array([int(x) if x is not None else -1 for x in sem_id]) == gt_cat
        )
        known_to_novel = known_mask & (ptype == "novel")
        known_to_wrong_known = known_mask & (ptype == "known") & ~known_correct
        known_unresolved = known_mask & (ptype == "unresolved")
        supported_mask = np.array(
            [g["protocol_role"] == "supported_known" for _, g, _ in rows]
        )
        zero_mask = np.array([g["protocol_role"] == "zero_shot_known" for _, g, _ in rows])
        supported_known_acc = float(known_correct[supported_mask].mean()) if supported_mask.sum() else 0.0
        zero_shot_known_acc = float(known_correct[zero_mask].mean()) if zero_mask.sum() else 0.0
        overall_known_acc = float(known_correct[known_mask].mean()) if n_known else 0.0
        known_to_novel_error = float(known_to_novel[known_mask].mean()) if n_known else 0.0
        known_misclassification_rate = float(known_to_wrong_known[known_mask].mean()) if n_known else 0.0
        known_unresolved_rate = float(known_unresolved[known_mask].mean()) if n_known else 0.0

        # ---- Novel routing ----
        routed = novel_mask & (ptype == "novel")
        false_known = novel_mask & (ptype == "known")
        unresolved_novel = novel_mask & (ptype == "unresolved")
        novel_routing_recall = float(routed[novel_mask].mean()) if n_novel else 0.0
        novel_type_pred = ptype == "novel"
        novel_routing_precision = (
            float((novel_type_pred & novel_mask).sum() / novel_type_pred.sum())
            if novel_type_pred.sum()
            else 0.0
        )
        false_known_absorption_rate = float(false_known[novel_mask].mean()) if n_novel else 0.0
        unresolved_novel_rate = float(unresolved_novel[novel_mask].mean()) if n_novel else 0.0

        # ---- Novel Hungarian (routed only) ----
        pred_virt = np.array([int(x) if x is not None else -1 for x in virt_id])
        routed_rows = np.where(routed)[0]
        pred_ids = sorted(set(int(pred_virt[i]) for i in routed_rows))
        gt_novel_cats = sorted(set(int(gt_cat[i]) for i in np.where(novel_mask)[0]))
        pred_index = {p: i for i, p in enumerate(pred_ids)}
        gt_index = {c: j for j, c in enumerate(gt_novel_cats)}
        W = np.zeros((len(pred_ids), len(gt_novel_cats)), dtype=np.int64)
        for i in routed_rows:
            W[pred_index[int(pred_virt[i])], gt_index[int(gt_cat[i])]] += 1
        assignment = {}
        if W.size:
            rows_, cols_ = linear_sum_assignment(-W)
            assignment = {
                pred_ids[r]: gt_novel_cats[c] for r, c in zip(rows_, cols_)
            }
        correct_novel = np.zeros(len(rows), dtype=bool)
        for i in routed_rows:
            if assignment.get(int(pred_virt[i])) == int(gt_cat[i]):
                correct_novel[i] = True
        route_aware_novel_acc = float(correct_novel[novel_mask].mean()) if n_novel else 0.0
        conditional_novel_acc = (
            float(correct_novel[routed].mean()) if routed.sum() else 0.0
        )

        # macro novel class accuracy (route-aware)
        macro_accs = []
        for c in gt_novel_cats:
            mask = novel_mask & (gt_cat == c)
            num = int((correct_novel & mask).sum())
            den = int(mask.sum())
            macro_accs.append(num / den if den else 0.0)
        macro_novel_class_acc = float(np.mean(macro_accs)) if macro_accs else 0.0

        # novel-only NMI/ARI on routed tracks
        if routed.sum() > 1:
            y_n = np.array([int(gt_cat[i]) for i in routed_rows])
            p_n = np.array([int(pred_virt[i]) for i in routed_rows])
            novel_nmi = float(normalized_mutual_info_score(y_n, p_n))
            novel_ari = float(adjusted_rand_score(y_n, p_n))
        else:
            novel_nmi = novel_ari = 0.0

        # fragmentation / merge / duplicate (routed novel tracks)
        frag = defaultdict(set)
        for i in routed_rows:
            frag[int(gt_cat[i])].add(int(pred_virt[i]))
        mean_fragmentation = float(np.mean([len(v) for v in frag.values()])) if frag else 0.0
        used_virt = defaultdict(set)
        for i in routed_rows:
            used_virt[int(pred_virt[i])].add(int(gt_cat[i]))
        merge_error = (
            float(sum(1 for s in used_virt.values() if len(s) > 1) / len(used_virt))
            if used_virt
            else 0.0
        )
        dup_cats = {c: v for c, v in frag.items() if len(v) > 1}
        duplicate_creation_rate = float(len(dup_cats) / len(frag)) if frag else 0.0
        duplicate_avg_extra = (
            float(np.mean([len(v) - 1 for v in dup_cats.values()])) if dup_cats else 0.0
        )

        # assignment delay: first correct virtual assignment per GT novel category
        first_true = {}
        first_correct = {}
        for i in np.where(novel_mask)[0]:
            c = int(gt_cat[i])
            first_true.setdefault(c, int(stream_order[i]))
            if correct_novel[i]:
                first_correct.setdefault(c, int(stream_order[i]))
        delays = [
            first_correct[c] - first_true[c]
            for c in first_true
            if c in first_correct
        ]
        mean_assignment_delay = float(np.mean(delays)) if delays else None

        predicted_novel_count = len(pred_ids)
        true_novel_count = len(gt_novel_cats)
        count_error = abs(predicted_novel_count - true_novel_count)

        # overall
        all_correct = known_correct | correct_novel
        all_track_acc = float(all_correct.mean()) if len(rows) else 0.0
        if overall_known_acc + route_aware_novel_acc > 0:
            harmonic = 2.0 * overall_known_acc * route_aware_novel_acc / (
                overall_known_acc + route_aware_novel_acc
            )
        else:
            harmonic = 0.0

        return {
            "num_samples": int(len(rows)),
            "num_known": n_known,
            "num_novel": n_novel,
            "num_novel_categories": true_novel_count,
            "supported_known_acc": supported_known_acc,
            "zero_shot_known_acc": zero_shot_known_acc,
            "overall_known_acc": overall_known_acc,
            "known_to_novel_error": known_to_novel_error,
            "known_misclassification_rate": known_misclassification_rate,
            "known_unresolved_rate": known_unresolved_rate,
            "novel_routing_recall": novel_routing_recall,
            "novel_routing_precision": novel_routing_precision,
            "false_known_absorption_rate": false_known_absorption_rate,
            "unresolved_novel_rate": unresolved_novel_rate,
            "route_aware_novel_acc": route_aware_novel_acc,
            "conditional_novel_acc": conditional_novel_acc,
            "novel_only_nmi": novel_nmi,
            "novel_only_ari": novel_ari,
            "macro_novel_class_acc": macro_novel_class_acc,
            "predicted_novel_count": predicted_novel_count,
            "novel_count_abs_error": count_error,
            "mean_fragmentation": mean_fragmentation,
            "merge_error": merge_error,
            "duplicate_creation_rate": duplicate_creation_rate,
            "duplicate_avg_extra": duplicate_avg_extra,
            "mean_assignment_delay": mean_assignment_delay,
            "all_track_acc": all_track_acc,
            "macro_known_novel_harmonic": harmonic,
            "memory_size": (metadata or {}).get("memory_size"),
            "inference_time_s": (metadata or {}).get("inference_time_s"),
            "hungarian_assignment": assignment,
        }


def load_gt_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
