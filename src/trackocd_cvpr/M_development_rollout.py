"""Complete causal frozen M/state-EMA streams, EXTERNAL version outcome audit.

No asset loading, fitting, GT feedback or checkpoint choice. Caller must gate
all9 actual M fits/source/authority/cache BEFORE private label/input loaders.
"""
from collections import Counter
import hashlib
import numpy as np
import torch
from .reliability_member_control import ReliabilityMemberControl
from .frame_atomic_state_ema import StateEMAMemberControl
from .versioned_causal_outcome_audit import VersionedCausalOutcomeAudit
from .policy_training_data import policy_current_frames
from .C_development_rollout import raw_fixed_scores


def readonly_M_stream(videos, labels, known, known_bindings, cfg, model_name, order_seed,
                      guard, *, model=None, rename_and_reverse=False):
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    if model_name not in ('Mean_latest', 'StateEMA_latest', 'relative_quarantine',
                          'relative_weighted', 'pooled_weighted'):
        raise ValueError('Only preregistered M families')
    baseline = model_name in ('Mean_latest', 'StateEMA_latest')
    if baseline != (model is None): raise ValueError('Frozen baseline has zero fit parameters; trained M requires own endpoint')
    if model is not None and (model.training or any(p.requires_grad for p in model.parameters())):
        raise ValueError('Strict frozen M endpoint')
    before = None if model is None else digest(model)
    order = np.random.default_rng(order_seed).permutation(len(videos)).tolist()
    order_hash = hashlib.sha256(np.asarray(order, dtype='<i8').tobytes()).hexdigest()
    routes = {}
    if rename_and_reverse:
        for video, arrays in videos:
            routes[video] = {int(t): 'M-renamed-' + str(len(arrays['track_id']) - i - 1) for i, t in enumerate(arrays['track_id'])}
        labels = {(v, routes[v][int(t)]): label for (v, t), label in labels.items()}
    def score(f, c): return raw_fixed_scores(f, c, **cfg['raw_fixed_thresholds'])
    limits = {'max_members': cfg['physical_tracks'], 'max_tokens': cfg['observations'],
              'max_commits': cfg['observations'], 'max_payload_bytes': 67108864}
    if baseline:
        control = StateEMAMemberControl(known, score, K=8, momentum=0. if model_name == 'Mean_latest' else .9,
                                       max_versions=cfg['max_versions'], bank_limits=limits)
    else:
        control = ReliabilityMemberControl(known, score, model, K=8, quarantine=model_name == 'relative_quarantine',
                                           max_versions=cfg['max_versions'], bank_limits=limits)
    audit = VersionedCausalOutcomeAudit(labels, known_bindings, prefixes=tuple(cfg['prefix_snapshots']),
        max_versions=cfg['max_versions'], max_commits=cfg['observations'])
    frames = empty = observed = 0; actions = Counter(); private = []
    with torch.inference_mode():
        for position, ordinal in enumerate(order):
            video, arrays = videos[ordinal]
            if position: control.start_next_video()
            audit.start_video(position, video)
            for frame, rows in policy_current_frames(arrays, routing_bindings=routes.get(video)):
                if rename_and_reverse: rows = tuple(reversed(rows))
                guard(); result = control.ingest(frame, rows)
                bank, _ = control.state_snapshot()
                audit.audit_frame(video, result, control.chosen_version_bindings(), control.memory_versions(), bank.commit_history())
                frames += 1; empty += int(not rows); observed += len(rows)
                for aliases, event in zip(result.routing_groups, result.events): actions[event.action] += len(aliases)
                private.append((video, result, control.chosen_version_bindings()))
    if (frames, empty, observed) != (cfg['metadata_frames'], cfg['empty_frames'], cfg['observations']):
        raise ValueError('Complete fixed physical clock, no short/unknown/empty drop')
    if model is not None and digest(model) != before: raise ValueError('Frozen M model changed')
    return {**audit.finish(), 'order_seed': order_seed, 'private_video_ordinal_order_sha256': order_hash,
        'metadata_frames': frames, 'empty_frames': empty, 'observations': observed, 'physical_actions': dict(actions),
        'optimizer_steps': 0, 'model_state_unchanged': True, 'GT_oracle_control_not_Pred': True}, tuple(private)
