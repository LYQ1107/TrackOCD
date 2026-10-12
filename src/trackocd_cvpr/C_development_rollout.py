"""Complete frozen-scorer GT-control causal streams, external truth audit only.

No optimizer, original annotation/pixels or Pred claim. Caller must gate actual
source/permission/cache/all9fit endpoints BEFORE invoking private loaders.
"""
from collections import Counter
import hashlib
import numpy as np
import torch

from .R_development_evaluation import load_completed_dev_tracks
from .policy_training_data import policy_current_frames
from .atomic_member_control import AtomicMemberControl
from .causal_category_audit import CausalCategoryAudit
from .candidate_scientific_training import variant_numeric_features


def dev_videos_and_external_labels(root, cfg, guard):
    tracks, labels, proof = load_completed_dev_tracks(root, cfg, guard)
    videos = {}; external = {}
    for track, label in zip(tracks, labels):
        if track.video in videos and videos[track.video] is not track.arrays:
            raise ValueError('One complete verified payload per video')
        videos[track.video] = track.arrays
        key = (track.video, str(int(track.arrays['track_id'][track.position])))
        if key in external: raise ValueError('Unique complete external physical GT keys')
        external[key] = label
    if len(videos) != 72 or len(external) != 472: raise ValueError('Full uncapped development scope')
    return list(videos.items()), external, proof


def raw_fixed_scores(features, context, *, tau_known, tau_existing):
    """Inherited nearest arbitration, NOT old update/ID-tie/function parity.

    Existing if above its gate and >= bestKnown, else Known above its gate,
    else NEW. Exact maximum/group-binding ambiguity WAIT via common resolver.
    Same frozen .65/.55 gates, no dev grid or labels. Both Raw K controls use
    this function; K expansion alone cannot rerank the maximum-cosine row.
    """
    known = features[:, 1] == 1; existing = ~known; cosine = features[:, 0]
    bk = float(cosine[known].max()) if known.any() else -1.
    be = float(cosine[existing].max()) if existing.any() else -1.
    scores = np.full(len(features) + 2, -2., dtype=np.float32)
    if existing.any() and be >= tau_existing and be >= bk:
        scores[np.flatnonzero(existing & (cosine == be))] = 1.
    elif known.any() and bk >= tau_known:
        scores[np.flatnonzero(known & (cosine == bk))] = 1.
    else:
        scores[-2] = 1.
    return scores


def readonly_C_stream(videos, labels, known_prototypes, known_bindings, cfg,
                      model_name, order_seed, guard, *, model=None, rename_and_reverse=False):
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    raw = model_name in ('Raw_Top1', 'Raw_Top8')
    if raw != (model is None): raise ValueError('Raw has no learned checkpoint; trained variants require frozen model')
    if model is not None and (model.training or any(p.requires_grad for p in model.parameters())):
        raise ValueError('Strict frozen scorer, no training in development')
    before = digest(model) if model is not None else None
    order = np.random.default_rng(order_seed).permutation(len(videos)).tolist()
    order_hash = hashlib.sha256(np.asarray(order, dtype='<i8').tobytes()).hexdigest()
    routes = {}
    if rename_and_reverse:
        for video, arrays in videos:
            routes[video] = {int(t): 'renamed-' + str(len(arrays['track_id']) - i - 1)
                             for i, t in enumerate(arrays['track_id'])}
        labels = {(v, routes[v][int(t)]): label for (v, t), label in labels.items()}
    def score(features, context):
        if raw:
            return raw_fixed_scores(features, context, **cfg['raw_fixed_thresholds'])
        with torch.inference_mode():
            return model(torch.tensor(variant_numeric_features(features, model_name)), torch.tensor(context.copy())).numpy().copy()
    K = cfg['K_per_family'][model_name]
    control = AtomicMemberControl(known_prototypes, score, K=K,
        bank_limits={'max_members': cfg['physical_tracks'], 'max_tokens': cfg['observations'],
                     'max_commits': cfg['observations'], 'max_payload_bytes': 67108864})
    audit = CausalCategoryAudit(labels, known_bindings, prefixes=tuple(cfg['prefix_snapshots']), max_tokens=cfg['observations'])
    frames = empty = observed = 0; actions = Counter(); private = []
    for position, ordinal in enumerate(order):
        video, arrays = videos[ordinal]
        if position: control.start_next_video()  # Preserve category memory, reset only current-video router.
        for frame, observations in policy_current_frames(arrays, routing_bindings=routes.get(video)):
            if rename_and_reverse: observations = tuple(reversed(observations))
            guard(); result = control.ingest(frame, observations)
            audit.audit_frame(video, result)  # AFTER own predictions; no return to policy.
            frames += 1; empty += int(not observations); observed += len(observations)
            for aliases, event in zip(result.routing_groups, result.events): actions[event.action] += len(aliases)
            private.append((video, result))
    if (frames, empty, observed) != (cfg['metadata_frames'], cfg['empty_frames'], cfg['observations']):
        raise ValueError('No lost physical observations/empty frames or successful subset')
    if model is not None and digest(model) != before: raise ValueError('Read-only model changed')
    result = audit.finish()
    return {**result, 'order_seed': order_seed, 'private_video_ordinal_order_sha256': order_hash,
        'metadata_frames': frames, 'empty_frames': empty, 'physical_actions': dict(actions),
        'optimizer_steps': 0, 'model_state_unchanged': True, 'GT_oracle_control_not_Pred': True}, tuple(private)
