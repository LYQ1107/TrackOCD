"""Bounded fit-Known optimization plumbing, NOT novel/memory evaluation.

Training supervision and video exclusion stay outside numeric model.forward.
Offline fit-only Known prototypes may use complete other-video training
tracks. Query prefixes are offline augmentation, not a frame-online policy.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from .differentiable_support_bank import VisiblePrefix
from .exact_support_reduction import reencode_exact_supports
from .GT_video_cache import read_verified_shard, observed_prefix
from .fit_loss_supervision import project_fit_loss_targets
from .raw_preserving_evidence import raw_geometry_loss
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import fit_video_ids
from src.trackocd_v2.io import sha256_file


@dataclass(frozen=True)
class FitTrack:
    video: int
    arrays: dict
    position: int
    loss_target: int | None

    def count(self):
        return int(self.arrays['offsets'][self.position + 1] - self.arrays['offsets'][self.position])

    def visible(self, observed):
        view = observed_prefix(self.arrays, self.position, observed)
        return VisiblePrefix(torch.from_numpy(view.visual.copy()), torch.from_numpy(view.boxes_normalized_xyxy.copy()),
                             torch.from_numpy(view.quality.copy()), torch.from_numpy(view.elapsed_frames.copy()))


def load_completed_fit_tracks(root, cfg, guard):
    root = Path(root)
    directory = root / cfg['fit_cache_directory']
    result = json.loads((directory / 'result.json').read_text())
    terminal = json.loads(directory.with_name(directory.name + '_supervisor_attempt_1.json').read_text())
    if (result['status'] != 'VERIFIED_FIT_PARTITION_UNCAPPED_GT_CONTROL_FEATURE_CACHE_NOT_PRED_OR_TRAINED'
            or result['live_remote_registration_proof']['sha'] != cfg['fit_cache_execution_commit']
            or (result['videos'], result['tracks'], result['observations'], result['metadata_frames']) != (257, 1401, 28326, 9345)
            or not result['frozen_parameters_unchanged'] or result['optimizer_or_formal_training_started']
            or result['frozen_parameter_sha256'] != '42c4c180dbaa825ee5bfecc6e6bf392af6439a1d2aa97f05095ac09b5e7097a9'
            or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] is not None):
        raise ValueError('Actual full fit-only cache / owned terminal qualification required before fitting')
    frozen = json.loads((root / cfg['frozen_protocol']).read_text())
    paths = {'private_assignment': 'private_global_assignment', 'private_Known_support': 'private_Known_support'}
    for path, key in paths.items():
        if (root / cfg[path]).is_symlink() or sha256_file(root / cfg[path]) != frozen['input_sha256'][key]:
            raise ValueError('Registered assignment / existing Known targets changed')
    bindings, targets, proof = project_fit_loss_targets(root / cfg['private_assignment'], root / cfg['private_Known_support'])
    fit = fit_video_ids(root / cfg['private_assignment'])
    index = json.loads((directory / 'private_shard_index.json').read_text())
    if (len(index) != 257 or {r['video_id'] for r in index} != fit
            or sorted(r['ordinal'] for r in index) != list(range(257))):
        raise ValueError('All fixed fit videos, no role expansion or missing rows')
    source_directories = {'outputs/trackocd_cvpr/cvpr2/pilot_GT_video_cache_v1', cfg['fit_cache_directory']}
    public_shards = {r['ordinal']: r for r in result['shards']}
    tracks, observed_known, joined = [], 0, set()
    for row in sorted(index, key=lambda r: r['ordinal']):
        guard()
        if row['source_directory'] not in source_directories:
            raise ValueError('Only verified original-pilot or new bounded fit shard')
        arrays, record = read_verified_shard(root / row['source_directory'], row['ordinal'])
        if (record['lineage']['opaque_video_id'] != row['video_id']
                or record['lineage']['frozen_protocol_sha256'] != sha256_file(root / cfg['frozen_protocol'])
                or record['lineage']['physical_index_sha256'] != frozen['input_sha256']['private_physical_GT']
                or record['lineage']['image_plan_sha256'] != frozen['input_sha256']['private_image_plan']
                or record['lineage']['checkpoint_sha256'] != frozen['GT_control_feature_IO']['checkpoint_sha256']
                or record['payload']['sha256'] != row['payload_sha256']
                or record['payload']['sha256'] != public_shards[row['ordinal']]['private_payload_sha256']):
            raise ValueError('Complete cache routing/hash/frozen-control lineage differs')
        for position, track in enumerate(arrays['track_id']):
            key = (row['video_id'], int(track)); target = targets.get(key)
            if target is not None:
                joined.add(key); observed_known += int(arrays['offsets'][position + 1] - arrays['offsets'][position])
            tracks.append(FitTrack(row['video_id'], arrays, position, target))
    if (len(tracks) != 1401 or len(joined) != len(targets) or observed_known != proof['eligible_supervised_observations']
            or observed_known != 22279 or sum(t.count() for t in tracks) != 28326):
        raise ValueError('Complete physical / loss-target denominator reconciliation')
    return tracks, {**proof, 'physical_tracks_kept': len(tracks), 'masked_no_fit5_target_tracks': 243,
                    'eligible_supervised_observations_joined': observed_known, 'raw_category_bindings_private': True}


def support_groups(tracks, query_video, class_count):
    groups = [[] for _ in range(class_count)]
    seen = [set() for _ in range(class_count)]
    for track in tracks:
        if track.loss_target is None or track.video == query_video:
            continue
        prefix = track.visible(track.count())
        digest = hashlib.sha256()
        for tensor in (prefix.visual, prefix.boxes, prefix.quality, prefix.elapsed):
            array = tensor.numpy(); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
        first, end = track.arrays['offsets'][track.position:track.position + 2]
        absolute_frames = track.arrays['frame_index'][first:end].tobytes()
        alias = (track.video, absolute_frames, digest.digest())  # same absolute frame scope, not identity feature
        if alias not in seen[track.loss_target]:
            groups[track.loss_target].append(prefix); seen[track.loss_target].add(alias)
    if any(not group for group in groups):
        raise ValueError('No queryself/samevideo fallback for missing Known support')
    return groups


def current_R_fit_loss(model, query_prefix, groups, external_target, *, temperature, geometry_weight, consistency_weight):
    if type(external_target) is not int or not 0 <= external_target < len(groups):
        raise ValueError('External eligible Known loss target, never unknown/background')
    if not np.isfinite(temperature) or temperature <= 0 or any(not np.isfinite(w) or w < 0 for w in (geometry_weight, consistency_weight)):
        raise ValueError('Finite positive temperature and nonnegative fixed loss weights')
    raw, adapted, statistics, proof = reencode_exact_supports(model, groups, torch.ones(len(groups)), torch.ones(len(groups)))
    query = model(query_prefix.visual, query_prefix.boxes, query_prefix.quality, query_prefix.elapsed)
    first = model(query_prefix.visual[:1], query_prefix.boxes[:1], query_prefix.quality[:1], query_prefix.elapsed[:1])
    adapted.retain_grad(); query.embedding.retain_grad()
    logits = adapted @ query.embedding / temperature
    CE = torch.nn.functional.cross_entropy(logits[None], torch.tensor([external_target]))
    geometry = raw_geometry_loss(torch.cat((raw, query.raw[None])), torch.cat((adapted, query.embedding[None])))
    consistency = (1 - (query.embedding * first.embedding).sum()).clamp(0, 2)
    loss = CE + geometry_weight * geometry + consistency_weight * consistency
    return loss, {'Known_CE': CE, 'Raw_Gram': geometry, 'same_track_past_prefix': consistency}, adapted, query.embedding, proof


def bounded_fit_steps(model, tracks, cfg, guard, progress):
    rng = np.random.default_rng(cfg['seed'])
    classes = cfg['class_count']
    by_class = [[t for t in tracks if t.loss_target == c] for c in range(classes)]
    if any(not rows for rows in by_class):
        raise ValueError('All fixed fit5 classes required, no successful subset')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    curve = []
    for step in range(cfg['optimizer_steps']):
        guard(); c = step % classes
        query = by_class[c][int(rng.integers(len(by_class[c])))]
        # Offline supervised augmentation only; full source length never goes
        # into numerical forward or a claim of deployment causal scheduling.
        observed = int(rng.integers(1, query.count() + 1))
        groups = support_groups(tracks, query.video, classes)
        optimizer.zero_grad(set_to_none=True)
        loss, terms, support, query_embedding, proof = current_R_fit_loss(model, query.visible(observed), groups, c,
            temperature=cfg['temperature'], geometry_weight=cfg['geometry_weight'], consistency_weight=cfg['consistency_weight'])
        if not torch.isfinite(loss): raise ValueError('Finite preregistered fit loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        if (not torch.isfinite(grad) or support.grad is None or query_embedding.grad is None
                or support.grad.norm() <= 0 or query_embedding.grad.norm() <= 0
                or model.residual[-1].weight.grad.norm() <= 0):
            raise ValueError('Real query/current-weight support gradients required')
        optimizer.step(); guard()
        row = {'step': step + 1, 'loss': float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()},
               'gradient_norm_before_clip': float(grad), 'support_gradient_norm': float(support.grad.norm()),
               'query_gradient_norm': float(query_embedding.grad.norm()),
               'support_groups': proof['groups_reencoded'], 'support_observations': proof['observations_reencoded']}
        curve.append(row); progress(row)
    return optimizer, curve, rng.bit_generator.state


def scientific_R_steps(model, tracks, window, variant, seed, guard, progress, save_checkpoint, *, resume=None, previous_curve=()):
    """Fixed endpoint seeded R trial; no dev/final/GT memory repair inputs.

    One representation direction, NOT three distinct scientific mechanisms.
    All support member embeddings, not just prototype averages, receive the
    Raw teacher geometry constraint. Query scheduling remains offline fit.
    """
    from .batched_evidence import reencode_batched_supports
    setting = window['variants'][variant]; cfg = window['training']
    rng = np.random.default_rng(seed)
    by_class = [[t for t in tracks if t.loss_target == c] for c in range(cfg['fit_supervised_classes'])]
    if any(not rows for rows in by_class): raise ValueError('All fixed fit classes, no favorable subset')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    curve = list(previous_curve); start = 0
    if resume is not None:
        start = resume['step']
        if (resume['seed'] != seed or resume['variant'] != variant or len(curve) != start
                or not 0 < start < cfg['steps_per_variant_and_seed']):
            raise ValueError('Exact prior variant/seed/step/curve, no silent fresh restart')
        model.load_state_dict(resume['model_state'], strict=True)
        optimizer.load_state_dict(resume['optimizer_state'])
        rng.bit_generator.state = resume['numpy_rng_state']; torch.set_rng_state(resume['torch_rng_state'])
    for step in range(start, cfg['steps_per_variant_and_seed']):
        guard(); target = step % len(by_class)
        query_track = by_class[target][int(rng.integers(len(by_class[target])))]
        observed = int(rng.integers(1, query_track.count() + 1))
        groups = support_groups(tracks, query_track.video, len(by_class))
        optimizer.zero_grad(set_to_none=True)
        raw, adapted, _, members = reencode_batched_supports(model, groups, torch.ones(len(groups)), torch.ones(len(groups)))
        prefix = query_track.visible(observed)
        query = model(prefix.visual, prefix.boxes, prefix.quality, prefix.elapsed)
        first = model(prefix.visual[:1], prefix.boxes[:1], prefix.quality[:1], prefix.elapsed[:1])
        adapted.retain_grad(); query.embedding.retain_grad()
        CE = torch.nn.functional.cross_entropy((adapted @ query.embedding / cfg['temperature'])[None], torch.tensor([target]))
        geometry = raw_geometry_loss(torch.cat((members.raw, query.raw[None])),
                                     torch.cat((members.embedding, query.embedding[None])))
        consistency = (1 - (query.embedding * first.embedding).sum()).clamp(0, 2)
        loss = CE + setting['geometry_weight'] * geometry + setting['consistency_weight'] * consistency
        if not torch.isfinite(loss): raise ValueError('Finite fixed scientific loss')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        if (not torch.isfinite(norm) or adapted.grad.norm() <= 0 or query.embedding.grad.norm() <= 0):
            raise ValueError('Finite real currentR support/query gradients')
        optimizer.step(); guard()
        row = {'step': step + 1, 'loss': float(loss.detach()), 'Known_CE': float(CE.detach()),
               'all_member_Raw_Gram': float(geometry.detach()), 'past_prefix_consistency': float(consistency.detach()),
               'gradient_norm_before_clip': float(norm), 'support_members': len(members.raw),
               'support_observations': sum(len(p.quality) for group in groups for p in group)}
        curve.append(row); progress(row, curve)
        if step + 1 in cfg['checkpoint_steps']:
            save_checkpoint(model, optimizer, step + 1, rng.bit_generator.state)
    return curve
