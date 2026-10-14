"""Causal joint R/C pilot with prediction-owned support lineage.

This module is a bounded integration pilot, not a qualified discovery result.
The controller uses the existing anonymous router/bank lifecycle, while the
candidate score is computed from tensor evidence so a single loss can reach
both the representation and candidate scorer.  Labels are consulted only
after an actual frame decision for the external loss; they never enter the
numeric forward, candidate selection, bank write, or token allocation.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from copy import deepcopy
import struct

import numpy as np
import torch

from .atomic_member_control import MemberControlEvent, MemberFrameResult
from .batched_evidence import reencode_batched_supports
from .candidate_score_contract import build_candidates, resolve_scores
from .candidate_training_loss import (
    ExternalPredictionHistory,
    candidate_supervised_loss,
    lawful_label,
    prediction_owned_loss_plan,
)
from .causal_observation_router import CurrentObservationRouter
from .differentiable_candidate_features import candidate_tensors
from .differentiable_support_bank import VisiblePrefix
from .prefix_member_bank import PrefixMemberBank
from .raw_preserving_evidence import RawPreservingEvidence
from .candidate_score_contract import CandidateScoreMLP


def visible_payload_to_prefix(payload):
    """Decode one bank-produced visible payload without IDs or semantic data."""
    if not isinstance(payload, bytes) or len(payload) < 8:
        raise ValueError('Bank-produced visible payload required')
    (count,) = struct.unpack_from('<Q', payload, 0)
    if count < 1:
        raise ValueError('Nonempty visible prefix required')
    offset = 8
    sizes = (count * 768 * 4, count * 4 * 4, count * 4, count * 8)
    end = offset + sum(sizes)
    if len(payload) != end:
        raise ValueError('Exact visible payload layout required')
    visual = np.frombuffer(payload, dtype='<f4', count=count * 768, offset=offset).reshape(count, 768).copy()
    offset += sizes[0]
    boxes = np.frombuffer(payload, dtype='<f4', count=count * 4, offset=offset).reshape(count, 4).copy()
    offset += sizes[1]
    quality = np.frombuffer(payload, dtype='<f4', count=count, offset=offset).copy()
    offset += sizes[2]
    elapsed = np.frombuffer(payload, dtype='<i8', count=count, offset=offset).copy()
    if (not np.isfinite(visual).all() or not np.isfinite(boxes).all() or not np.isfinite(quality).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any((quality < 0) | (quality > 1))
            or elapsed[0] != 0 or np.any(elapsed[1:] <= elapsed[:-1])):
        raise ValueError('Invalid bank-produced visible payload')
    return VisiblePrefix(torch.from_numpy(visual), torch.from_numpy(boxes),
                         torch.from_numpy(quality), torch.from_numpy(elapsed))


def prefix_view_to_tensor(view):
    """Make a detached numeric prefix from a current router view."""
    return VisiblePrefix(
        torch.from_numpy(np.asarray(view.visual, dtype=np.float32).copy()),
        torch.from_numpy(np.asarray(view.boxes_normalized_xyxy, dtype=np.float32).copy()),
        torch.from_numpy(np.asarray(view.quality, dtype=np.float32).copy()),
        torch.from_numpy(np.asarray(view.elapsed_frames, dtype=np.int64).copy()),
    )


def _class_macro_loss(group_losses):
    if not group_losses:
        raise ValueError('At least one eligible joint loss family required')
    values = [torch.stack(terms).sort().values.mean() for terms in group_losses.values() if terms]
    if not values:
        raise ValueError('At least one eligible joint loss term required')
    return torch.stack(values).sort().values.mean()


@dataclass(frozen=True, slots=True)
class JointFrame:
    result: MemberFrameResult
    logits: tuple[torch.Tensor, ...]


class JointAtomicMemberControl:
    """Existing anonymous controller with a differentiable score callback.

    Raw candidate selection and discrete actions are detached bookkeeping.  The
    returned logits retain the current query/support graph for the caller's
    post-decision external loss.  No routing ID, token, or label is passed to
    either neural module.
    """

    def __init__(self, evidence, scorer, known_prototypes, known_support_groups, *, K=8,
                 max_candidate_groups=4096, max_tracks=4096, max_observations=65536,
                 bank_limits=None):
        if not isinstance(evidence, RawPreservingEvidence) or not isinstance(scorer, CandidateScoreMLP):
            raise ValueError('Registered tensor evidence and joint scorer required')
        if scorer.feature_count != 14 or type(K) is not int or K not in (1, 3, 5, 8):
            raise ValueError('Fixed 14-feature joint scorer and K required')
        keys = tuple(sorted(known_prototypes))
        if keys != tuple(range(len(keys))) or len(keys) != len(known_support_groups):
            raise ValueError('Dense external Known prototype/support bindings required')
        if any(not group for group in known_support_groups):
            raise ValueError('Nonempty lawful Known support groups required')
        self.evidence = evidence
        self.scorer = scorer
        self._known = {key: np.asarray(known_prototypes[key], dtype=np.float32).copy() for key in keys}
        self._known_support_groups = tuple(tuple(group) for group in known_support_groups)
        self._router = CurrentObservationRouter(max_tracks=max_tracks, max_observations=max_observations)
        self._bank = PrefixMemberBank(**(bank_limits or {}))
        self._K = K
        self._candidate_bound = max_candidate_groups
        self._next_id = 0

    def start_next_video(self):
        bank = deepcopy(self._bank)
        bank.start_next_video()
        self._bank = bank
        self._router = CurrentObservationRouter(max_tracks=self._router.max_tracks,
                                                max_observations=self._router.max_observations)

    def _support_by_token(self):
        result = {}
        for support in self._bank.raw_support_snapshot():
            result[support.token] = tuple(visible_payload_to_prefix(payload) for payload in support.payloads)
        return result

    def _candidate_supports(self, bindings, by_token):
        """Expand one feature-indistinguishable binding group canonically."""
        payloads = {}
        kinds = set()
        for binding in bindings:
            kinds.add(binding.kind)
            if binding.kind == 'KNOWN':
                identifier = binding.identifier
                if type(identifier) is not int or not 0 <= identifier < len(self._known_support_groups):
                    raise ValueError('External Known binding outside fixed support set')
                group = self._known_support_groups[identifier]
            elif binding.kind == 'EXISTING':
                if not isinstance(binding.identifier, str) or binding.identifier not in by_token:
                    raise ValueError('Existing binding lacks actual bank support')
                group = by_token[binding.identifier]
            else:
                raise ValueError('Unexpected candidate binding kind')
            for prefix in group:
                if not isinstance(prefix, VisiblePrefix):
                    raise ValueError('Numeric visible support required')
                # Payload bytes are the canonical anonymous support identity;
                # do not use a routing or semantic ID to break ties.
                key = tuple(x.detach().cpu().numpy().astype('<f4').tobytes()
                            for x in (prefix.visual, prefix.boxes, prefix.quality)) + (
                                prefix.elapsed.detach().cpu().numpy().astype('<i8').tobytes(),)
                payloads[key] = prefix
        if len(kinds) != 1 or not payloads:
            raise ValueError('Nonempty homogeneous candidate support group required')
        return tuple(payloads[key] for key in sorted(payloads))

    def ingest(self, frame_index, observations, *, track_grad=True):
        router = deepcopy(self._router)
        bank = deepcopy(self._bank)
        batch = router.ingest(frame_index, observations)
        prior = bank.state_evidence()
        by_token = {support.token: tuple(visible_payload_to_prefix(payload)
                                         for payload in support.payloads)
                    for support in bank.raw_support_snapshot()}
        events = []
        assignments = []
        graphs = []
        next_id = self._next_id
        for view, aliases, signature in zip(batch.model_views, batch.routing_groups, batch.visible_input_sha256):
            snapshot = build_candidates(view, self._known, prior, K=self._K,
                                        max_groups=self._candidate_bound)
            query = prefix_view_to_tensor(view)
            groups = [self._candidate_supports(bindings, by_token) for bindings in snapshot.bindings]
            if groups:
                flags = torch.tensor([float(bindings[0].kind == 'KNOWN') for bindings in snapshot.bindings],
                                     dtype=torch.float32)
                binding_counts = torch.tensor([len(bindings) for bindings in snapshot.bindings],
                                              dtype=torch.float32)
                raw, adapted, stats, _ = reencode_batched_supports(
                    self.evidence, groups, flags, binding_counts)
            else:
                raw = torch.empty((0, 768), dtype=torch.float32)
                adapted = torch.empty((0, 768), dtype=torch.float32)
                stats = torch.empty((0, 5), dtype=torch.float32)
            with torch.set_grad_enabled(track_grad):
                query_evidence = self.evidence(query.visual, query.boxes, query.quality, query.elapsed)
                rows, context = candidate_tensors(query_evidence, raw, adapted, stats)
                logits = self.scorer(rows, context)
            scores = np.asarray(logits.detach().cpu().numpy(), dtype='<f4').copy()
            action, binding, reason = resolve_scores(snapshot, scores)
            category = binding if action == 'KNOWN' else None
            token = binding if action == 'EXISTING' else None
            if action == 'NEW':
                token = f'S:{next_id}'
                next_id += 1
            assignments.append(token if action in ('NEW', 'EXISTING') else None)
            events.append(MemberControlEvent(
                signature,
                rows.detach().cpu().numpy().astype('<f4').tobytes(),
                context.detach().cpu().numpy().astype('<f4').tobytes(),
                snapshot.bindings,
                scores.tobytes(), action, category, token, reason))
            graphs.append(logits)
        bank.apply_frame(batch, assignments)
        result = MemberFrameResult(batch.frame_index, tuple(events), batch.routing_groups)
        self._router, self._bank, self._next_id = router, bank, next_id
        return JointFrame(result, tuple(graphs))


def train_joint_policy_pilot(videos, projection, known_prototypes, known_support_groups,
                             cfg, guard, progress, checkpoint=None):
    """Run a fixed small causal pilot over a preregistered Train subset.

    The complete selected pilot clock is always consumed.  ``projection`` is
    only converted to external post-decision targets here; no target reaches
    ``JointAtomicMemberControl``.  This function deliberately returns a pilot
    receipt summary, not development metrics or a selected checkpoint.
    """
    if not videos or len({video for video, _ in videos}) != len(videos):
        raise ValueError('Unique fixed pilot videos required')
    if checkpoint is not None and not callable(checkpoint):
        raise ValueError('Checkpoint callback must be callable')
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    evidence = RawPreservingEvidence().train()
    scorer = CandidateScoreMLP(feature_count=14).train()
    optimizer = torch.optim.AdamW(tuple(evidence.parameters()) + tuple(scorer.parameters()),
                                  lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    external_targets = {(video, str(track)): target for (video, track), target in projection.targets.items()}
    curve = []
    episode_summaries = []
    total_steps = 0
    for episode in range(cfg['episodes']):
        order = rng.permutation(len(videos)).tolist()
        controller = JointAtomicMemberControl(evidence, scorer, known_prototypes,
            known_support_groups, K=cfg['K_per_family'],
            bank_limits={'max_members': cfg['physical_tracks_bound'],
                         'max_tokens': cfg['observations_bound'],
                         'max_commits': cfg['observations_bound'],
                         'max_payload_bytes': cfg['payload_bytes_bound']})
        history = ExternalPredictionHistory(max_commits=cfg['observations_bound'])
        steps = 0; frames = empty = observations = groups = 0
        actions = Counter(); loss_reasons = Counter(); families = Counter()
        sealed = []
        for position, ordinal in enumerate(order):
            video, arrays = videos[ordinal]
            if position:
                controller.start_next_video()
            from .policy_training_data import policy_current_frames
            for frame, rows in policy_current_frames(arrays):
                guard()
                prior = history.snapshot()
                joint = controller.ingest(frame, rows, track_grad=steps < cfg['updates_per_episode'])
                terms = defaultdict(list)
                other_member = other_video = 0
                for logits, aliases, event in zip(joint.logits, joint.result.routing_groups, joint.result.events):
                    keys = tuple((video, route) for route in aliases)
                    plan = prediction_owned_loss_plan(event.candidate_bindings_for_external_loss,
                                                      keys, external_targets, prior)
                    loss = candidate_supervised_loss(logits, plan)
                    loss_reasons[plan.reason] += 1
                    if loss is not None and steps < cfg['updates_per_episode']:
                        label = lawful_label(external_targets.get(keys[0], ('no_retained_semantic_truth', None)))
                        terms[label].append(loss)
                        families[plan.query_family] += 1
                        other_member += int(plan.positive_has_other_physical_member)
                        other_video += int(plan.positive_has_other_video_member)
                if terms and steps < cfg['updates_per_episode']:
                    loss = _class_macro_loss(terms)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(tuple(evidence.parameters()) + tuple(scorer.parameters()),
                                                          cfg['gradient_clip'])
                    evidence_grad = sum(float(p.grad.detach().norm()) for p in evidence.parameters()
                                        if p.grad is not None)
                    scorer_grad = sum(float(p.grad.detach().norm()) for p in scorer.parameters()
                                      if p.grad is not None)
                    if (not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0
                            or evidence_grad <= 0 or scorer_grad <= 0):
                        raise ValueError('Real joint R/C gradients required')
                    optimizer.step(); steps += 1; total_steps += 1
                    row = {'step': total_steps, 'episode': episode, 'episode_optimizer_step': steps,
                           'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
                           'evidence_gradient_norm_sum': evidence_grad, 'scorer_gradient_norm_sum': scorer_grad,
                           'supervised_families_this_frame': dict(families),
                           'positive_other_physical_groups_this_frame': other_member,
                           'positive_other_video_groups_this_frame': other_video}
                    curve.append(row); progress(row, curve)
                    if checkpoint is not None and total_steps in cfg['checkpoint_steps']:
                        checkpoint(evidence, scorer, optimizer, total_steps, episode, steps,
                                  rng.bit_generator.state)
                history.observe_frame(video, joint.result, external_targets)
                for aliases, event in zip(joint.result.routing_groups, joint.result.events):
                    actions[event.action] += len(aliases)
                sealed.append((video, joint.result))
                frames += 1; empty += int(not rows); observations += len(rows); groups += len(joint.result.events)
                guard()
        if steps != cfg['updates_per_episode']:
            raise ValueError('Fixed joint pilot optimizer count unmet')
        episode_summaries.append({
            'episode': episode, 'optimizer_steps': steps, 'videos': len(videos),
            'metadata_frames': frames, 'empty_frames': empty, 'physical_observations': observations,
            'canonical_groups': groups, 'physical_actions': dict(actions),
            'loss_plan_reasons': dict(loss_reasons), 'eligible_loss_families': dict(families),
            'prediction_owned_commit_records': len(history.records()),
            'complete_clock_consumed': True,
        })
    return evidence, scorer, optimizer, {
        'optimizer_steps': total_steps,
        'trainable_parameters': sum(p.numel() for p in evidence.parameters()) + sum(p.numel() for p in scorer.parameters()),
        'episodes': episode_summaries,
        'loss_curve': curve,
        'all_selected_pilot_frames_and_observations_consumed': True,
        'joint_graph_reached_R_and_C': True,
        'prediction_owned_memory_only': True,
        'GT_or_id_or_token_in_numeric_forward': False,
        'dev_final_Val_Test_or_new_nonKnown_semantics_read': False,
        'development_or_goal_acceptance': False,
    }, tuple(sealed)
