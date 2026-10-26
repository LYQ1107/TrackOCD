"""Joint R/C/M causal integration on prediction-owned support.

This is an integration layer for the next registered pilot.  The discrete
versioned-memory lifecycle remains the existing prospective, append-only
contract.  For candidate scoring, each visible support is re-encoded by R and
its adapted members are weighted by M before the 14-feature C bridge is built.
External M agreement loss is still computed only after the actual pre-frame
memory snapshot and predictions are available.  No labels, IDs or tokens enter
any neural forward or lifecycle write.
"""

from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import torch

from .atomic_member_control import MemberControlEvent, MemberFrameResult
from .candidate_score_contract import build_candidates, resolve_scores
from .candidate_score_contract import CandidateScoreMLP
from .candidate_training_loss import (
    ExternalPredictionHistory,
    candidate_supervised_loss,
    lawful_label,
    prediction_owned_loss_plan,
)
from .causal_observation_router import CurrentObservationRouter
from .differentiable_candidate_features import candidate_tensors
from .differentiable_support_bank import VisiblePrefix
from .joint_rc_policy_training import (
    JointAtomicMemberControl,
    _class_macro_loss,
    prefix_view_to_tensor,
    visible_payload_to_prefix,
)
from .member_reliability import MemberReliabilityMLP, member_features
from .policy_training_data import policy_current_frames
from .raw_preserving_evidence import RawPreservingEvidence, unit
from .reliability_training_loss import (
    prediction_owned_support_targets,
    support_agreement_losses,
    support_agreement_plan,
)
from .versioned_reliability_memory import VersionedReliabilityMemory
from .version_lineage_audit import audit_version_lineage


def _reliable_group(evidence, member_model, group, known_flag, binding_count):
    if not group or any(type(prefix) is not VisiblePrefix for prefix in group):
        raise ValueError('Nonempty visible support group required')
    rows = []
    qualities = []
    counts = []
    for prefix in group:
        value = evidence(prefix.visual.clone(), prefix.boxes.clone(),
                         prefix.quality.clone(), prefix.elapsed.clone())
        rows.append(value)
        qualities.append(value.quality_mean)
        counts.append(len(prefix.quality))
    raw_rows = torch.stack([value.raw for value in rows])
    adapted_rows = torch.stack([value.embedding for value in rows])
    raw = unit(raw_rows.mean(0))
    quality = torch.stack(qualities)
    observed = torch.tensor(counts, dtype=torch.float32, device=raw.device)
    features, vectors = member_features(adapted_rows, quality, observed)
    logits = member_model(features)
    weights = 1e-3 + (1 - 1e-3) * logits.sigmoid()
    adapted = unit((vectors * weights[:, None]).sum(0))
    dispersion = (1 - (raw_rows @ raw).mean()).clamp(0, 1)
    stats = torch.stack((raw.new_tensor(float(known_flag)), dispersion,
                         raw.new_tensor(float(len(group))), quality.mean(),
                         raw.new_tensor(float(binding_count))))
    return raw, adapted, stats


class JointRCMAtomicMemberControl(JointAtomicMemberControl):
    """Joint tensor candidate bridge plus existing prospective M lifecycle."""

    def __init__(self, evidence, scorer, member_model, known_prototypes, known_support_groups, *,
                 quarantine=False, max_versions=65536, K=8, **kwargs):
        if not isinstance(member_model, MemberReliabilityMLP):
            raise ValueError('Registered M reliability module required')
        super().__init__(evidence, scorer, known_prototypes, known_support_groups, K=K, **kwargs)
        self.member_model = member_model
        self._memory = VersionedReliabilityMemory(quarantine=quarantine, max_versions=max_versions)
        self._preframe_numeric = {}
        self._chosen_versions = ()

    def memory_versions(self):
        return self._memory.versions()

    def preframe_training_evidence(self):
        return dict(self._preframe_numeric)

    def chosen_version_bindings(self):
        return self._chosen_versions

    def _candidate_supports_by_token(self, bank):
        return {support.token: tuple(visible_payload_to_prefix(payload)
                                     for payload in support.payloads)
                for support in bank.raw_support_snapshot()}

    def start_next_video(self):
        super().start_next_video()
        self._preframe_numeric = {}
        self._chosen_versions = ()

    def state_snapshot(self):
        """Expose the immutable pre-frame bank handle to external loss code."""
        return deepcopy(self._bank), self._next_id

    def ingest(self, frame_index, observations, *, track_grad=True):
        router = deepcopy(self._router)
        bank = deepcopy(self._bank)
        memory = deepcopy(self._memory)
        batch = router.ingest(frame_index, observations)
        prior, numeric = memory.refresh(bank, self.member_model)
        versions = memory.current_bindings()
        by_token = self._candidate_supports_by_token(bank)
        known = {key: np.frombuffer(value, dtype=np.float32) for key, value in self._known.items()}
        events = []
        assignments = []
        chosen = []
        graphs = []
        next_id = self._next_id
        for view, signature in zip(batch.model_views, batch.visible_input_sha256):
            snapshot = build_candidates(view, known, prior, K=self._K,
                                        max_groups=self._candidate_bound)
            query = prefix_view_to_tensor(view)
            groups = [self._candidate_supports(bindings, by_token) for bindings in snapshot.bindings]
            if groups:
                raw_rows = []
                adapted_rows = []
                stats = []
                for bindings, group in zip(snapshot.bindings, groups):
                    raw, adapted, row_stats = _reliable_group(
                        self.evidence, self.member_model, group,
                        float(bindings[0].kind == 'KNOWN'), len(bindings))
                    raw_rows.append(raw); adapted_rows.append(adapted); stats.append(row_stats)
                raw = torch.stack(raw_rows)
                adapted = torch.stack(adapted_rows)
                stats = torch.stack(stats)
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
            chosen.append(versions.get(token) if action == 'EXISTING' else None)
            events.append(MemberControlEvent(
                signature, rows.detach().cpu().numpy().astype('<f4').tobytes(),
                context.detach().cpu().numpy().astype('<f4').tobytes(),
                snapshot.bindings, scores.tobytes(), action, category, token, reason))
            graphs.append(logits)
        bank.apply_frame(batch, assignments)
        memory.refresh(bank, self.member_model)
        result = MemberFrameResult(batch.frame_index, tuple(events), batch.routing_groups)
        self._router, self._bank, self._memory, self._next_id = router, bank, memory, next_id
        self._preframe_numeric, self._chosen_versions = numeric, tuple(chosen)
        return result, tuple(graphs)


def train_joint_rcm_policy_pilot(videos, projection, known_prototypes, known_support_groups,
                                 cfg, guard, progress, checkpoint=None, scorer_factory=None):
    """Run a fixed causal R/C/M pilot; no dev metrics or endpoint selection."""
    if not videos or len({video for video, _ in videos}) != len(videos):
        raise ValueError('Unique fixed R/C/M pilot videos required')
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    evidence = RawPreservingEvidence().train()
    if scorer_factory is None:
        scorer = CandidateScoreMLP(feature_count=14).train()
    else:
        scorer = scorer_factory().train()
    member = MemberReliabilityMLP(pooled_control=False).train()
    parameters = tuple(evidence.parameters()) + tuple(scorer.parameters()) + tuple(member.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    external_targets = {(video, str(track)): target for (video, track), target in projection.targets.items()}
    curve = []
    episode_summaries = []
    total_steps = 0
    all_gradients = Counter()
    for episode in range(cfg['episodes']):
        order = rng.permutation(len(videos)).tolist()
        controller = JointRCMAtomicMemberControl(
            evidence, scorer, member, known_prototypes, known_support_groups,
            quarantine=cfg['quarantine'], max_versions=cfg['max_versions'], K=cfg['K_per_family'],
            bank_limits={'max_members': cfg['physical_tracks_bound'],
                         'max_tokens': cfg['observations_bound'],
                         'max_commits': cfg['observations_bound'],
                         'max_payload_bytes': cfg['payload_bytes_bound']})
        history = ExternalPredictionHistory(max_commits=cfg['observations_bound'])
        epochs = {}
        steps = frames = empty = observations = groups = 0
        actions = Counter(); C_reasons = Counter(); M_reasons = Counter(); families = Counter()
        versions_at_end = ()
        for position, ordinal in enumerate(order):
            video, arrays = videos[ordinal]
            epochs[position] = video
            if position:
                controller.start_next_video()
            for frame, rows in policy_current_frames(arrays):
                guard()
                prior_bank, _ = controller.state_snapshot()
                prior_history = history.snapshot()
                result, logits_rows = controller.ingest(frame, rows,
                                                         track_grad=steps < cfg['updates_per_episode'])
                C_terms = defaultdict(list)
                M_terms = defaultdict(list)
                other_member = other_video = 0
                for logits, aliases, event in zip(logits_rows, result.routing_groups, result.events):
                    keys = tuple((video, route) for route in aliases)
                    plan = prediction_owned_loss_plan(event.candidate_bindings_for_external_loss,
                                                      keys, external_targets, prior_history)
                    loss = candidate_supervised_loss(logits, plan)
                    C_reasons[plan.reason] += 1
                    if loss is not None and steps < cfg['updates_per_episode']:
                        label = lawful_label(external_targets.get(keys[0], ('no_retained_semantic_truth', None)))
                        C_terms[label].append(loss)
                        families['C_' + plan.query_family] += 1
                        other_member += int(plan.positive_has_other_physical_member)
                        other_video += int(plan.positive_has_other_video_member)
                for token, reliability in controller.preframe_training_evidence().items():
                    labels = prediction_owned_support_targets(prior_bank, token, external_targets, epochs)
                    plan = support_agreement_plan(labels)
                    M_reasons[plan.reason] += 1
                    for label, loss in support_agreement_losses(reliability.logits, plan):
                        if steps < cfg['updates_per_episode']:
                            M_terms[label].append(loss)
                            families['M_' + label[0]] += 1
                if (C_terms or M_terms) and steps < cfg['updates_per_episode']:
                    C_loss = _class_macro_loss(C_terms) if C_terms else None
                    M_loss = _class_macro_loss(M_terms) if M_terms else None
                    loss = C_loss if C_loss is not None else torch.zeros((), dtype=torch.float32)
                    if M_loss is not None:
                        loss = loss + cfg['memory_loss_weight'] * M_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'])
                    grad_sums = {
                        'R': sum(float(p.grad.detach().norm()) for p in evidence.parameters() if p.grad is not None),
                        'C': sum(float(p.grad.detach().norm()) for p in scorer.parameters() if p.grad is not None),
                        'M': sum(float(p.grad.detach().norm()) for p in member.parameters() if p.grad is not None),
                    }
                    if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
                        raise ValueError('Finite joint R/C/M optimizer gradient required')
                    if min(grad_sums.values()) <= 0:
                        raise ValueError('Every joint R/C/M branch must receive a real gradient')
                    optimizer.step(); steps += 1; total_steps += 1
                    all_gradients.update({key: int(value > 0) for key, value in grad_sums.items()})
                    row = {'step': total_steps, 'episode': episode, 'episode_optimizer_step': steps,
                           'loss': float(loss.detach()),
                           'C_loss': None if C_loss is None else float(C_loss.detach()),
                           'M_loss': None if M_loss is None else float(M_loss.detach()),
                           'gradient_norm_before_clip': float(norm),
                           'R_gradient_norm_sum': grad_sums['R'], 'C_gradient_norm_sum': grad_sums['C'],
                           'M_gradient_norm_sum': grad_sums['M'],
                           'positive_other_physical_groups_this_frame': other_member,
                           'positive_other_video_groups_this_frame': other_video}
                    curve.append(row); progress(row, curve)
                    if checkpoint is not None and total_steps in cfg['checkpoint_steps']:
                        checkpoint(evidence, scorer, member, optimizer, total_steps, episode, steps,
                                  rng.bit_generator.state)
                history.observe_frame(video, result, external_targets)
                for aliases, event in zip(result.routing_groups, result.events):
                    actions[event.action] += len(aliases)
                frames += 1; empty += int(not rows); observations += len(rows); groups += len(result.events)
                guard()
        if steps != cfg['updates_per_episode']:
            raise ValueError('Fixed joint R/C/M optimizer count unmet')
        bank, _ = controller.state_snapshot()
        versions_at_end = controller.memory_versions()
        lineage = audit_version_lineage(versions_at_end, bank.commit_history(),
                                        external_targets, epochs,
                                        max_union_references=cfg['max_union_references'])
        episode_summaries.append({
            'episode': episode, 'optimizer_steps': steps, 'videos': len(videos),
            'metadata_frames': frames, 'empty_frames': empty, 'physical_observations': observations,
            'canonical_groups': groups, 'physical_actions': dict(actions),
            'C_loss_plan_reasons': dict(C_reasons), 'M_loss_plan_reasons': dict(M_reasons),
            'eligible_optimizer_families': dict(families),
            'prediction_owned_commit_records': len(history.records()),
            'version_records': len(versions_at_end), 'append_only_memory_history_retained': True,
            'version_lineage_audit': lineage,
            'complete_clock_consumed': True,
        })
    if min(all_gradients.get(key, 0) for key in ('R', 'C', 'M')) <= 0:
        raise ValueError('All joint branch gradient evidence must be retained')
    return evidence, scorer, member, optimizer, {
        'optimizer_steps': total_steps, 'trainable_parameters': sum(p.numel() for p in parameters),
        'episodes': episode_summaries, 'loss_curve': curve,
        'joint_graph_reached_R_C_and_M': True, 'prediction_owned_memory_only': True,
        'GT_or_id_or_token_in_numeric_forward': False,
        'dev_final_Val_Test_or_new_nonKnown_semantics_read': False,
        'development_or_goal_acceptance': False,
    }
