"""Frozen checkpoint diagnostics; predictions and GT auditing are separated.

Oracle variants are explicitly controlled Train-only diagnostic upper bounds,
not deployable methods. The unmodified step accepts visual inputs only.
"""
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from src.trackocd_core.persistent_policy import (
    ACTIONS, CandidateMemory, SupervisionHistory, masked_logits, predicted_action,
)
from src.trackocd_core.train_first_replay import causal_rows
from src.trackocd_core.evaluation import DecisionEvent, TrackKey, seal_decisions


def visual_step(model, memory, item, wait_bias):
    cues, k, t = memory.candidates(item['embedding'], item['uncertainty'],
                                  item['maturity'], item['quality'], item['elapsed'])
    with torch.inference_mode():
        logits = masked_logits(model, torch.tensor(cues))
    return predicted_action(logits, wait_bias), k, t, cues, logits.numpy().copy()


def ranked_candidates(memory, item):
    z = np.asarray(item['embedding'], dtype=np.float32)
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    known = sorted(((float(z @ v), k) for k, v in memory.known.items()), key=lambda x: (-x[0], x[1]))
    anonymous = sorted(((float(z @ m.prototype), t) for t, m in memory.anonymous.items()),
                       key=lambda x: (-x[0], x[1]))
    return [k for _, k in known], [t for _, t in anonymous]


def diagnostic_replay(routes, order, prefix, proto, bank, model, labels, constants,
                      wait_bias, intervention='ACTUAL_FROZEN_CHECKPOINT', keep_trace=False):
    if intervention not in {'ACTUAL_FROZEN_CHECKPOINT', 'ORACLE_CANDIDATE_TOP8',
                            'ORACLE_ACTION_TOP1', 'ORACLE_ISOLATE_FIRST_BAD_WRITE'}:
        raise ValueError('Unregistered diagnostic intervention')
    memory = CandidateMemory(proto)
    audit = SupervisionHistory()  # external GT targets, never a visual_step argument
    past_videos = defaultdict(set)
    events, trace = [], []
    counts, candidates, targets, confusion = Counter(), Counter(), Counter(), Counter()
    isolated = False
    first_bad = None
    delayed_risk_events = 0
    for seq, route in enumerate(causal_rows(routes, order, prefix)):
        item = bank[route['key']]
        action, k, t, features, logits = visual_step(model, memory, item, wait_bias)
        # All label-dependent operations occur AFTER the unchanged visual prediction.
        label = labels[route['key']]
        category = label['category_id']
        is_known = label['simulation_role'] == 'known'
        kr, ar = ranked_candidates(memory, item)
        pure_cross_video = [token for token in ar if audit.pure(token, category) and
                            bool(past_videos[token] - {route['video_id']})] if not is_known else []
        known_rank = kr.index(category) + 1 if is_known and category in kr else None
        anon_rank = min((ar.index(token) + 1 for token in pure_cross_video), default=None)
        rank = known_rank if is_known else anon_rank
        role = 'known' if is_known else 'pseudo_novel'
        candidates[f'{role}_queries'] += 1
        candidates[f'{role}_correct_available'] += rank is not None
        for topk in (1, 3, 5, 8):
            candidates[f'{role}_top{topk}_hit'] += rank is not None and rank <= topk
        costs, target, credit = audit.costs(category, is_known, k, t, item['actual_prefix'], constants)
        targets[ACTIONS[target]] += 1
        confusion[f'{ACTIONS[target]}->{ACTIONS[action]}'] += 1
        delayed_risk_events += credit is not None
        original_action = action
        if intervention == 'ORACLE_CANDIDATE_TOP8':
            if action == 0 and known_rank is not None and known_rank <= 8:
                k = category
            elif action == 1 and anon_rank is not None and anon_rank <= 8:
                t = pure_cross_video[0]
        elif intervention == 'ORACLE_ACTION_TOP1':
            action = target
        bad_write = (action in (1, 2) and is_known) or (
            action == 1 and any(c != category for c in audit.members.get(t, [])))
        if bad_write and first_bad is None:
            first_bad = seq
        if intervention == 'ORACLE_ISOLATE_FIRST_BAD_WRITE' and bad_write and not isolated:
            action = 3
            isolated = True
        prediction = memory.apply(action, k, t, item['embedding'], item['quality'])
        audit.update(prediction, category, seq)
        if prediction['kind'] in {'NEW', 'EXISTING'}:
            past_videos[prediction['token']].add(route['video_id'])
        counts[prediction['kind']] += 1
        counts[f'{role}_{prediction["kind"]}'] += 1
        if keep_trace:
            trace.append({'sequence': seq, 'physical_key': route['key'], 'features': features,
                          'logits': logits, 'action': original_action, 'actual_prefix': item['actual_prefix'],
                          'correct_known_rank': known_rank, 'correct_pure_cross_video_anonymous_rank': anon_rank,
                          'nominal_target': target, 'costs': costs, 'delayed_credit_sequence': credit})
        events.append(DecisionEvent(seq, TrackKey(route['video_id'], str(route['physical_track_id'])),
                                    item['actual_prefix'], prediction['kind'],
                                    known_category_id=prediction.get('category_id'), token=prediction.get('token')))
    sealed = seal_decisions(events, video_order=order, prefix_cap=prefix, known_ids=tuple(proto))
    return sealed, {
        'intervention': intervention, 'not_method_result': intervention != 'ACTUAL_FROZEN_CHECKPOINT',
        'counts': dict(counts), 'candidate_counts_same_real_history': dict(candidates),
        'nominal_targets': dict(targets), 'target_predicted_confusion_before_intervention': dict(confusion),
        'first_bad_write_sequence': first_bad, 'isolated_writes': int(isolated),
        'delayed_risk_events': delayed_risk_events,
        'anonymous_states': len(memory.anonymous),
        'mixed_states': sum(len(set(v)) > 1 for v in audit.members.values()),
        'commitment_coverage': 1 - counts['WAIT'] / max(len(routes), 1),
        'independent_capped_prefix_not_dense_online': True,
        'oracle_interventions_change_future_history': intervention != 'ACTUAL_FROZEN_CHECKPOINT',
    }, trace


def geometry_diagnostics(vectors, rows, raw_vectors):
    vectors = np.asarray(vectors, np.float64)
    raw_vectors = np.asarray(raw_vectors, np.float64)
    category = np.asarray([r['category_id'] for r in rows])
    video = np.asarray([r['video_id'] for r in rows])
    allowed = video[:, None] != video[None, :]
    same = category[:, None] == category[None, :]
    valid = (same & allowed).any(1)
    cosine = vectors @ vectors.T
    raw_cosine = raw_vectors @ raw_vectors.T
    hit = same[np.arange(len(rows)), np.where(allowed, cosine, -np.inf).argmax(1)]
    raw_hit = same[np.arange(len(rows)), np.where(allowed, raw_cosine, -np.inf).argmax(1)]
    spectra = {}
    for centered in (False, True):
        z = vectors - vectors.mean(0) if centered else vectors
        s = np.linalg.svd(z, compute_uv=False)
        eigen = s * s
        p = eigen / max(eigen.sum(), 1e-12)
        nonzero = p[p > 0]
        spectra['centered' if centered else 'uncentered'] = {
            'effective_rank': float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
            'top_eigenvalue_share': float(p[0]), 'spectrum': eigen.tolist(),
        }
    positive = cosine[same & allowed]
    negative = cosine[~same & allowed]
    hard_negative = np.where(~same & allowed, cosine, -np.inf).max(1)
    results = {'queries': len(rows), 'spectra': spectra,
               'mean_vector_norm_anisotropy': float(np.linalg.norm(vectors.mean(0))),
               'cross_video_positive_cosine_mean': float(positive.mean()) if positive.size else None,
               'cross_video_negative_cosine_mean': float(negative.mean()) if negative.size else None,
               'hard_negative_cosine_mean': float(hard_negative.mean()),
               'supported_queries': int(valid.sum()),
               'errors_corrected_vs_raw': int((valid & ~raw_hit & hit).sum()),
               'new_errors_vs_raw': int((valid & raw_hit & ~hit).sum())}
    for role in ('known', 'pseudo_novel'):
        role_mask = valid & np.asarray([r['simulation_role'] == role for r in rows])
        per_category = [float(hit[role_mask & (category == c)].mean()) for c in set(category[role_mask])]
        results[f'{role}_macro_rank1'] = float(np.mean(per_category)) if per_category else None
        results[f'{role}_supported'] = int(role_mask.sum())
    return results


def checkpoint_gradient_replay(model, routes, order, prefix, proto, bank, labels, constants,
                               category_weights, wait_bias, expected_weight, delayed_weight):
    """Read-only backward at a selected checkpoint, not historical gradients."""
    model.requires_grad_(True)
    memory, audit = CandidateMemory(proto), SupervisionHistory()
    ce_terms, expected_terms, delayed_terms, probabilities, actions = [], [], [], [], []
    for seq, route in enumerate(causal_rows(routes, order, prefix)):
        item = bank[route['key']]
        cues, k, t = memory.candidates(item['embedding'], item['uncertainty'], item['maturity'],
                                      item['quality'], item['elapsed'])
        logits = masked_logits(model, torch.tensor(cues))
        prob = torch.softmax(logits, -1)
        action = predicted_action(logits, wait_bias)
        label = labels[route['key']]
        costs, target, credit = audit.costs(label['category_id'], label['simulation_role'] == 'known',
                                            k, t, item['actual_prefix'], constants)
        weight = category_weights[label['category_id']]
        ce_terms.append(F.cross_entropy(logits[None], torch.tensor([target])) * weight)
        expected_terms.append((prob * torch.tensor(costs)).sum() * expected_weight * weight)
        if credit is not None:
            delayed_terms.append(probabilities[credit][actions[credit]] * delayed_weight *
                                 constants['later_predicted_memory_effect'] * weight)
        probabilities.append(prob)
        actions.append(action)
        audit.update(memory.apply(action, k, t, item['embedding'], item['quality']), label['category_id'], seq)
    losses = {'nominal_CE': torch.stack(ce_terms).mean(),
              'expected_cost': torch.stack(expected_terms).mean(),
              'delayed_credit': torch.stack(delayed_terms).sum() / len(routes) if delayed_terms else None}
    parameters = list(model.parameters())
    gradients, norms, magnitudes = {}, {}, {}
    for name, loss in losses.items():
        grad = torch.zeros(sum(p.numel() for p in parameters)) if loss is None else torch.cat([
            g.flatten() for g in torch.autograd.grad(loss, parameters, retain_graph=True)])
        gradients[name] = grad
        norms[name] = float(grad.norm())
        magnitudes[name] = float(loss.detach()) if loss is not None else 0.0
    cosines = {}
    for a, b in [('nominal_CE', 'expected_cost'), ('nominal_CE', 'delayed_credit'), ('expected_cost', 'delayed_credit')]:
        denom = norms[a] * norms[b]
        cosines[f'{a}__{b}'] = float(gradients[a] @ gradients[b] / denom) if denom > 1e-12 else None
    model.requires_grad_(False)
    return {'status': 'CHECKPOINT_GRADIENT_REPLAY_NOT_HISTORICAL_TRAIN_GRADIENT',
            'optimizer_steps': 0, 'loss_magnitudes': magnitudes, 'gradient_norms': norms,
            'gradient_cosines': cosines, 'delayed_events': len(delayed_terms),
            'wait_bias': wait_bias, 'prefix': prefix}
