"""Evaluator-only geometric reference joins; no full-track purity certificate."""
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.trackocd_core.physical_diagnostic import pairwise_iou


def truncated_join(gt_geometry, pred_geometry, threshold=.5, minimum_shared=2, minimum_fraction=.5):
    """Inputs: frame-index -> [(opaque track key, normalized xyxy), ...].

    No category/role/scores are accepted. All targets remain in denominators;
    unmatched predictions are unknown. A strict one-to-one vote assignment is
    evaluated after predictions seal, never used to prepare model input.
    """
    gt_counts, votes, pred_matched, observation_matches = Counter(), Counter(), Counter(), 0
    for frame, rows in gt_geometry.items():
        gt_counts.update(key for key, _ in rows)
        predicted = pred_geometry.get(frame, [])
        similarity = pairwise_iou([box for _, box in rows], [box for _, box in predicted])
        if not similarity.size:
            continue
        weights = np.where(similarity >= threshold, max(similarity.shape) + 1 + similarity, 0)
        g, p = linear_sum_assignment(weights, maximize=True)
        for a, b in zip(g, p):
            if weights[a, b] > 0:
                target, track = rows[a][0], predicted[b][0]
                votes[(target, track)] += 1
                pred_matched[track] += 1
                observation_matches += 1
    targets, tracks = sorted(gt_counts), sorted(pred_matched)
    weights = np.zeros((len(targets), len(tracks)), dtype=float)
    for i, target in enumerate(targets):
        for j, track in enumerate(tracks):
            n = votes[(target, track)]
            # Reliability pertains only to the selected capped reference observations.
            if n >= minimum_shared and n / gt_counts[target] >= minimum_fraction and n == pred_matched[track]:
                weights[i, j] = min(len(targets), len(tracks)) * max(gt_counts.values(), default=0) + 1 + n
    matches = {}
    if weights.size:
        g, p = linear_sum_assignment(weights, maximize=True)
        matches = {tracks[b]: targets[a] for a, b in zip(g, p) if weights[a, b] > 0}
    all_pred = {key for rows in pred_geometry.values() for key, _ in rows}
    return matches, {'GT_reference_targets': len(targets), 'predicted_tracks': len(all_pred),
        'observation_matches_to_truncated_reference': observation_matches,
        'canonical_reference_targets_matched': len(matches), 'unknown_unmatched_predicted_tracks': len(all_pred) - len(matches),
        'full_GT_or_global_track_purity_claim': False, 'matching_uses_category_or_role': False,
        'denominator_targets_removed': False, 'future_geometry_used_only_post_seal_not_policy': True}
