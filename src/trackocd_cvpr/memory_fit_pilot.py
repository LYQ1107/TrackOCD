"""Ten real M steps on prediction-owned policy support, after caller gates.

No GT placement/quarantine/reset; full rollout retained even when loss masked.
Pilot only, not three trained seeds, heldout gain or final memory acceptance.
"""
from collections import Counter, defaultdict
import torch
from .member_reliability import MemberReliabilityMLP
from .reliability_member_control import ReliabilityMemberControl
from .reliability_training_loss import (
    prediction_owned_support_targets, support_agreement_plan, support_agreement_losses)
from .candidate_training_loss import ExternalPredictionHistory
from .candidate_scientific_training import class_macro_loss
from .C_development_rollout import raw_fixed_scores
from .policy_training_data import policy_current_frames


def fit_memory_pilot(videos, projection, known, cfg, guard, progress):
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    torch.manual_seed(cfg['seed'])
    model = MemberReliabilityMLP().train()
    initial = digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    def score(f, c): return raw_fixed_scores(f, c, **cfg['raw_fixed_thresholds'])
    control = ReliabilityMemberControl(known, score, model, quarantine=True, K=8,
        max_versions=cfg['max_versions'], bank_limits={'max_members': 54, 'max_tokens': 1268,
        'max_commits': 1268, 'max_payload_bytes': 16777216})
    targets = {(v, str(t)): label for (v, t), label in projection.targets.items()}
    epochs = {}; history = ExternalPredictionHistory(max_commits=1268)
    frames = empty = observations = zero_gradient = 0
    actions = Counter(); reasons = Counter(); families = Counter(); classes = set()
    curve = []; sealed = []
    for position, (video, arrays) in enumerate(videos):
        epochs[position] = video
        if position: control.start_next_video()
        for frame, rows in policy_current_frames(arrays):
            guard(); prior, _ = control.state_snapshot()
            with torch.set_grad_enabled(len(curve) < cfg['optimizer_steps']):
                result = control.ingest(frame, rows)
            terms = defaultdict(list); frame_families = Counter(); scored_groups = masked_groups = 0
            for token, evidence in control.preframe_training_evidence().items():
                labels = prediction_owned_support_targets(prior, token, targets, epochs)
                plan = support_agreement_plan(labels); reasons[plan.reason] += 1
                losses = support_agreement_losses(evidence.logits, plan)
                masked_groups += len(labels) - len(losses)
                for label, loss in losses:
                    terms[label].append(loss); frame_families[label[0]] += 1; scored_groups += 1
            if terms and len(curve) < cfg['optimizer_steps']:
                loss = class_macro_loss(terms)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise ValueError('Nonfinite actual M optimizer loss/gradient')
                if norm <= 0:
                    zero_gradient += 1  # not an optimizer update or alternate target
                else:
                    optimizer.step(); families.update(frame_families); classes.update(terms)
                    row = {'step': len(curve) + 1, 'loss': float(loss.detach()),
                        'gradient_norm_before_clip': float(norm),
                        'actual_optimizer_family_groups': dict(frame_families),
                        'supervised_classes_this_update': len(terms), 'masked_support_groups_this_frame': masked_groups,
                        'eligible_support_groups_this_frame': scored_groups}
                    curve.append(row); progress(row, curve)
            history.observe_frame(video, result, targets)
            sealed.append((video, result, control.chosen_version_bindings()))
            frames += 1; empty += int(not rows); observations += len(rows)
            for aliases, event in zip(result.routing_groups, result.events): actions[event.action] += len(aliases)
            guard()
    summary = {'optimizer_steps': len(curve), 'trainable_parameters': 161,
        'model_parameter_sha256_before': initial, 'model_parameter_sha256_after': digest(model),
        'complete_rollout': {'videos': len(videos), 'metadata_frames': frames, 'empty_frames': empty,
                             'physical_observations': observations, 'physical_actions': dict(actions)},
        'actual_optimizer_family_groups': dict(families), 'actual_optimizer_semantic_classes': len(classes),
        'loss_plan_counts_all_frames': dict(reasons), 'zero_gradient_eligible_frames_without_update': zero_gradient,
        'memory_versions_retained': len(control.memory_versions()),
        'version_lifecycle_records': dict(Counter(v.lifecycle for v in control.memory_versions())),
        'actual_anonymous_commits_retained': len(history.records()), 'loss_curve': curve,
        'GT_driven_memory_repair_or_deleted_wrong_history': False,
        'strict_historical_purity_or_CT_or_Novel_gain_measured': False}
    if len(curve) != cfg['optimizer_steps'] or initial == digest(model):
        raise ValueError('Fixed ten actual M updates unmet; no favorable subset or alternate schedule')
    if (frames, empty, observations) != (374, 1, 1268):
        raise ValueError('Complete policy physical rollout required')
    model.eval(); optimizer.zero_grad(set_to_none=True)
    bank, _ = control.state_snapshot()
    return model, optimizer, summary, tuple(sealed), history.records(), control.memory_versions(), bank.commit_history()
