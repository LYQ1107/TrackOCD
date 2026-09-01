"""Category discovery evaluation metrics."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def hungarian_acc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    D = max(int(y_pred.max()), int(y_true.max())) + 1
    w = np.zeros((D, D), dtype=int)
    np.add.at(w, (y_pred, y_true), 1)
    rows, cols = linear_sum_assignment(w.max() - w)
    return w[rows, cols].sum() / len(y_true), w


def fragmentation_and_purity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    frag = {}
    for c in np.unique(y_true):
        mask = y_true == c
        frag[int(c)] = len(np.unique(y_pred[mask]))
    purity = {}
    for p in np.unique(y_pred):
        mask = y_pred == p
        counts = Counter(y_true[mask].tolist())
        purity[int(p)] = max(counts.values()) / mask.sum()
    return frag, purity


def merge_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_merged = 0
    for p in np.unique(y_pred):
        if len(np.unique(y_true[y_pred == p])) > 1:
            n_merged += 1
    return n_merged / len(np.unique(y_pred)) if len(np.unique(y_pred)) else 0.0


def duplicate_creation(pred_virtual_ids, y_true, known_mask):
    """Per true novel category: duplicate virtual ids assigned to it."""
    pred = np.asarray(pred_virtual_ids)
    y = np.asarray(y_true)
    mask = ~np.asarray(known_mask, dtype=bool)
    per_cat = defaultdict(set)
    for p, c in zip(pred[mask], y[mask]):
        per_cat[int(c)].add(int(p))
    dup_cats = {c: len(s) for c, s in per_cat.items() if len(s) > 1}
    rate = len(dup_cats) / len(per_cat) if per_cat else 0.0
    avg_extra = sum(v - 1 for v in dup_cats.values()) / len(per_cat) if per_cat else 0.0
    return rate, avg_extra, dict(dup_cats)


def evaluate_predictions(y_true, y_pred, known_mask, subset_ids=None, verbose=False):
    """Return dict of metrics. y_pred are virtual category ids (0..)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    known_mask = np.asarray(known_mask, dtype=bool)
    if subset_ids is not None:
        keep = np.isin(y_true, subset_ids)
        y_true, y_pred, known_mask = y_true[keep], y_pred[keep], known_mask[keep]
    # renumber predicted ids contiguously to keep metric matrices small
    unique_pred = np.unique(y_pred)
    remap = {int(p): i for i, p in enumerate(unique_pred)}
    y_pred = np.array([remap[int(p)] for p in y_pred])

    acc_all, w = hungarian_acc(y_true, y_pred)
    # V2-style known/novel ACC using global Hungarian mapping
    old_classes = set(y_true[known_mask].tolist())
    new_classes = set(y_true[~known_mask].tolist())
    rows, cols = linear_sum_assignment(w.max() - w)
    ind_map = {int(j): int(i) for i, j in zip(rows, cols)}

    def subset_acc(classes):
        if not classes:
            return 0.0
        num = 0
        den = 0
        for c in classes:
            gt_col = int(c)
            pred_row = ind_map[gt_col]
            num += w[pred_row, gt_col]
            den += w[:, gt_col].sum()
        return num / den if den else 0.0

    acc_known = subset_acc(old_classes)
    acc_novel = subset_acc(new_classes)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    frag, purity = fragmentation_and_purity(y_true, y_pred)
    merge = merge_error(y_true, y_pred)
    n_pred = len(np.unique(y_pred))
    n_true = len(np.unique(y_true))
    dup_rate, dup_avg_extra, dup_map = duplicate_creation(y_pred, y_true, known_mask)

    res = {
        "num_samples": int(len(y_true)),
        "acc_all": float(acc_all),
        "acc_known": float(acc_known),
        "acc_novel": float(acc_novel),
        "nmi": float(nmi),
        "ari": float(ari),
        "predicted_categories": int(n_pred),
        "true_categories": int(n_true),
        "category_count_abs_error": int(abs(n_pred - n_true)),
        "fragmentation": frag,
        "mean_fragmentation": float(np.mean(list(frag.values()))) if frag else 0.0,
        "purity": purity,
        "mean_purity": float(np.mean(list(purity.values()))) if purity else 0.0,
        "merge_error": float(merge),
        "duplicate_creation_rate": float(dup_rate),
        "duplicate_avg_extra": float(dup_avg_extra),
        "duplicate_map": dup_map,
    }
    if verbose:
        print(json.dumps(res, indent=1, default=str))
    return res


def load_private_labels(project_root):
    path = project_root / "data" / "tao_ow_ocd_v1" / "private" / "val_gt_track_labels.jsonl"
    labels = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            labels[r["sample_id"]] = r
    return labels
