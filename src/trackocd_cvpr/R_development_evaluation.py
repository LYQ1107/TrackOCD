"""Complete development GT-control embedding evaluation, never model fitting.

External labels only score outputs; no semantic/ID/role fields in R inputs.
This numerical evaluator cannot itself prove source/permission/cache lineage;
its registered worker must gate actual private reads before calling it.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .fit_R_training import FitTrack, load_completed_fit_tracks, support_groups
from .GT_video_cache import read_verified_shard, policy_dev_video_ids
from .batched_evidence import pack_visible, batched_evidence, reencode_batched_supports
from .R_development_metrics import project_dev_labels, reconcile_dev_labels, certified_cross_video_retrieval
from src.trackocd_v2.io import sha256_file


def load_completed_dev_tracks(root, cfg, guard):
    root = Path(root); directory = root / cfg['policy_dev_cache_directory']
    result = json.loads((directory / 'result.json').read_text())
    terminal = json.loads(directory.with_name(directory.name + '_supervisor_attempt_1.json').read_text())
    if (result['status'] != 'VERIFIED_POLICY_DEV_UNCAPPED_GT_CONTROL_CACHE_NOT_FINAL_OR_PRED_OR_TRAINED'
            or result['live_remote_registration_proof']['sha'] != cfg['policy_dev_execution_commit']
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None or result['independent_final_or_protected_pixels_read']
            or result['development_or_final_semantic_values_decoded']):
        raise ValueError('Completed qualified geometry-only policy/dev input and owned terminal proof')
    expected = {'videos': 72, 'tracks': 472, 'observations': 8753, 'metadata_frames': 2562}
    if any(result['role_totals']['development_pseudo'][k] != v for k, v in expected.items()):
        raise ValueError('Complete frozen development role denominator')
    frozen = json.loads((root / cfg['frozen_protocol']).read_text())
    for field, key in (('private_assignment', 'private_global_assignment'), ('private_Known_support', 'private_Known_support')):
        if (root / cfg[field]).is_symlink() or sha256_file(root / cfg[field]) != frozen['input_sha256'][key]:
            raise ValueError('Fixed authorized scalar label/assignment index')
    roles = policy_dev_video_ids(root / cfg['private_assignment'])
    dev = {v for v, role in roles.items() if role == 'development_pseudo'}
    projected, label_proof = project_dev_labels(root / cfg['private_assignment'], root / cfg['private_Known_support'])
    index = json.loads((directory / 'private_shard_index.json').read_text())
    if len(index) != 82 or {r['video_id'] for r in index} != set(roles):
        raise ValueError('Exact input video universe, no protected/final cache')
    public = {r['ordinal']: r for r in result['shards']}; tracks = []; keys = []; frame_count = 0
    for row in sorted(index, key=lambda r: r['ordinal']):
        if row['video_id'] not in dev: continue  # no policy geometry/label body read
        guard()
        if row['source_directory'] != cfg['policy_dev_cache_directory']:
            raise ValueError('No arbitrary alternate payload source')
        arrays, record = read_verified_shard(directory, row['ordinal'])
        lineage = record['lineage']
        if (lineage['opaque_video_id'] != row['video_id']
                or lineage['frozen_protocol_sha256'] != sha256_file(root / cfg['frozen_protocol'])
                or lineage['physical_index_sha256'] != frozen['input_sha256']['private_physical_GT']
                or lineage['image_plan_sha256'] != frozen['input_sha256']['private_image_plan']
                or lineage['checkpoint_sha256'] != frozen['GT_control_feature_IO']['checkpoint_sha256']
                or record['payload']['sha256'] != row['payload_sha256']
                or record['payload']['sha256'] != public[row['ordinal']]['private_payload_sha256']):
            raise ValueError('Private/public payload route and frozen oracle-control lineage')
        frame_count += len(arrays['clock_frame_index'])
        for i, identity in enumerate(arrays['track_id']):
            keys.append((row['video_id'], int(identity))); tracks.append(FitTrack(row['video_id'], arrays, i, None))
    labels, strata = reconcile_dev_labels(keys, projected)
    if sum(t.count() for t in tracks) != 8753 or frame_count != 2562:
        raise ValueError('Every annotated physical observation and metadata frame retained')
    return tracks, labels, {**label_proof, 'physical_label_strata': strata, 'physical_tracks': len(tracks),
                            'observations': 8753, 'metadata_frames': frame_count}


def known_prototypes(model, fit_tracks):
    # Legitimate offline fit5 supervision only; no development targets.
    groups = support_groups(fit_tracks, -1, 5)
    with torch.inference_mode():
        raw, adapted, _, _ = reencode_batched_supports(model, groups, torch.ones(5), torch.ones(5))
    return raw.numpy().copy(), adapted.numpy().copy()


def embed_complete_role(model, tracks, prefix_cap, guard):
    raw = []; adapted = []; actual = []
    with torch.inference_mode():
        for first in range(0, len(tracks), 128):
            guard(); selected = tracks[first:first + 128]
            counts = [t.count() if prefix_cap == 'all_available' else min(int(prefix_cap), t.count()) for t in selected]
            batch = batched_evidence(model, pack_visible([t.visible(n) for t, n in zip(selected, counts)]))
            raw.append(batch.raw.numpy().copy()); adapted.append(batch.embedding.numpy().copy()); actual.extend(counts)
    return np.concatenate(raw), np.concatenate(adapted), actual


def effective_rank(embeddings):
    s = np.linalg.svd(np.asarray(embeddings, np.float64), compute_uv=False)
    eigen = s * s; probability = eigen[eigen > 0] / eigen.sum()
    return float(np.exp(-(probability * np.log(probability)).sum()))


def known_recognition(embeddings, labels, fit_bindings, prototypes):
    e = np.asarray(embeddings, np.float64); p = np.asarray(prototypes, np.float64)
    e /= np.linalg.norm(e, axis=1, keepdims=True); p /= np.linalg.norm(p, axis=1, keepdims=True)
    scores = e @ p.T; per_class = {}; values = []
    for i, (kind, category) in enumerate(labels):
        if kind != 'Known': continue
        if category not in fit_bindings: raise ValueError('No dev-GT prototype or invented missing Known binding')
        ties = np.flatnonzero(scores[i] == scores[i].max())
        correct = float(fit_bindings[category] in ties) / len(ties)
        values.append(correct); per_class.setdefault(category, []).append(correct)
    if not values: raise ValueError('Complete frozen Known query population')
    return {'Known_queries': len(values), 'Known_classes_observed': len(per_class),
            'micro_accuracy_exact_tie_expectation': float(np.mean(values)),
            'macro_accuracy_exact_tie_expectation': float(np.mean([np.mean(v) for v in per_class.values()])),
            'prototype_source_offline_fit_only': True, 'not_open_set_or_full_semantic_Hscore': True}


def evaluate_prefix_embeddings(raw, adapted, tracks, labels, bindings, raw_prototypes, adapted_prototypes, actual):
    videos = [t.video for t in tracks]
    result = {}
    for name, e, p in (('Raw', raw, raw_prototypes), ('R', adapted, adapted_prototypes)):
        result[name] = {'retrieval': certified_cross_video_retrieval(e, videos, labels),
                        'Known': known_recognition(e, labels, bindings, p), 'effective_rank_full_physical': effective_rank(e)}
        for stratum, predicate in (('singleton', lambda t: t.count() == 1), ('short_le4', lambda t: t.count() <= 4)):
            scoped = [(kind if predicate(t) else 'gallery_only', label) for t, (kind, label) in zip(tracks, labels)]
            if any(kind == 'declared_pseudoNovel' for kind, _ in scoped):
                result[name][stratum + '_queries_full_gallery'] = certified_cross_video_retrieval(e, videos, scoped)
            else:
                result[name][stratum + '_queries_full_gallery'] = {'declared_query_tracks': 0, 'status': 'NO_QUERY_STRATUM_NOT_FILTERED_MAIN'}
    a = np.asarray(adapted, np.float64); r = np.asarray(raw, np.float64)
    cosine = (a * r).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1))
    result['Raw_drift'] = {'mean_cosine': float(cosine.mean()), 'minimum_cosine': float(cosine.min())}
    result['support'] = {'full_physical_tracks': len(tracks), 'visible_observations_forwarded': int(sum(actual)),
                         'singleton_tracks_kept': sum(t.count() == 1 for t in tracks),
                         'short_le4_tracks_kept': sum(t.count() <= 4 for t in tracks),
                         'full_physical_gallery_short_and_unlabeled_not_dropped': True}
    return result
