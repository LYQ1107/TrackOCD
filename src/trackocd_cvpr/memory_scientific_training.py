"""Fixed four-episode M controls on endogenous, prediction-owned Raw support.

No GT state placement/purity repair, dev labels, final selection or warmstart.
External lineage audit occurs after predictions and cannot feed policy/model.
"""
from collections import Counter, defaultdict
import numpy as np
import torch
from .member_reliability import MemberReliabilityMLP
from .reliability_member_control import ReliabilityMemberControl
from .reliability_training_loss import (
    prediction_owned_support_targets, support_agreement_plan, support_agreement_losses)
from .candidate_training_loss import ExternalPredictionHistory
from .candidate_scientific_training import class_macro_loss
from .C_development_rollout import raw_fixed_scores
from .policy_training_data import policy_current_frames
from .version_lineage_audit import audit_version_lineage


def scientific_memory_fit(videos, projection, known, cfg, variant, seed, guard, progress, checkpoint):
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    settings = cfg['variants'][variant]; training = cfg['training']
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = MemberReliabilityMLP(pooled_control=settings['pooled_control']).train()
    initial = digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    targets = {(v, str(t)): label for (v, t), label in projection.targets.items()}
    curve, summaries, private = [], [], []
    update_families, target_bins = Counter(), Counter(); classes, pseudo_classes = set(), set()
    def score(f, c): return raw_fixed_scores(f, c, **cfg['raw_fixed_thresholds'])
    for episode in range(training['episodes']):
        order = rng.permutation(len(videos)).tolist()
        control = ReliabilityMemberControl(known, score, model, quarantine=settings['quarantine'], K=8,
            max_versions=cfg['memory']['max_versions'], bank_limits={'max_members': 54, 'max_tokens': 1268,
            'max_commits': 1268, 'max_payload_bytes': 16777216})
        epochs = {}; history = ExternalPredictionHistory(max_commits=1268)
        frames = empty = observations = steps = zero_gradient = 0
        actions, reasons, opportunities, eligible_target_bins = Counter(), Counter(), Counter(), Counter()
        sealed = []
        for position, ordinal in enumerate(order):
            video, arrays = videos[ordinal]; epochs[position] = video
            if position: control.start_next_video()
            for frame, rows in policy_current_frames(arrays):
                guard(); prior, _ = control.state_snapshot()
                with torch.set_grad_enabled(steps < training['updates_per_episode']): result = control.ingest(frame, rows)
                terms = defaultdict(list); families, bins = Counter(), Counter(); masked = eligible = 0
                for token, evidence in control.preframe_training_evidence().items():
                    labels = prediction_owned_support_targets(prior, token, targets, epochs)
                    plan = support_agreement_plan(labels); reasons[plan.reason] += 1
                    losses = support_agreement_losses(evidence.logits, plan)
                    masked += len(labels) - len(losses)
                    for target in plan.targets:
                        if target is not None: bins['one' if target == 1 else 'zero' if target == 0 else 'fractional'] += 1
                    for label, loss in losses:
                        terms[label].append(loss); families[label[0]] += 1; eligible += 1
                opportunities.update(families); eligible_target_bins.update(bins)
                if terms and steps < training['updates_per_episode']:
                    loss = class_macro_loss(terms); optimizer.zero_grad(set_to_none=True); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'])
                    if not torch.isfinite(loss) or not torch.isfinite(norm): raise ValueError('Finite actual M optimizer gradient required')
                    if norm <= 0:
                        zero_gradient += 1
                    else:
                        optimizer.step(); steps += 1; update_families.update(families); target_bins.update(bins)
                        classes.update(terms); pseudo_classes.update(label for label in terms if label[0] == 'policy_pseudo')
                        row = {'step': len(curve) + 1, 'episode': episode, 'episode_optimizer_step': steps,
                            'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
                            'actual_optimizer_family_groups': dict(families), 'actual_optimizer_target_bins': dict(bins),
                            'supervised_classes_this_update': len(terms), 'masked_support_groups_this_frame': masked,
                            'eligible_support_groups_this_frame': eligible}
                        curve.append(row); progress(row, curve)
                        if len(curve) in training['checkpoint_steps']:
                            checkpoint(model, optimizer, len(curve), episode, steps, rng.bit_generator.state)
                history.observe_frame(video, result, targets)
                sealed.append((video, result, control.chosen_version_bindings()))
                frames += 1; empty += int(not rows); observations += len(rows)
                for aliases, event in zip(result.routing_groups, result.events): actions[event.action] += len(aliases)
                guard()
        bank, _ = control.state_snapshot(); versions, commits = control.memory_versions(), bank.commit_history()
        if (frames, empty, observations, len(videos)) != (374, 1, 1268, 10):
            raise ValueError('Every full physical policy episode, no favorable subset')
        if steps != training['updates_per_episode']:
            raise ValueError('Fixed50 actual M updates unmet, no alternate exposure order or seed')
        # AFTER all actual predictions, no callback into the controller/optimizer.
        strict = audit_version_lineage(versions, commits, targets, epochs)
        summaries.append({'episode': episode, 'optimizer_steps': steps, 'videos': len(videos),
            'metadata_frames': frames, 'empty_frames': empty, 'physical_observations': observations,
            'physical_actions': dict(actions), 'loss_plan_counts_all_frames': dict(reasons),
            'eligible_family_groups_all_frames_including_frozen_tail': dict(opportunities),
            'eligible_target_bins_all_frames_including_frozen_tail': dict(eligible_target_bins),
            'zero_gradient_eligible_frames_without_update': zero_gradient,
            'actual_anonymous_commits_retained': len(history.records()), 'actual_bank_commits_retained': len(commits),
            'memory_versions_retained': len(versions), 'version_lifecycle_records': dict(Counter(v.lifecycle for v in versions)),
            'external_strict_version_lineage': strict})
        private.append((order, tuple(sealed), history.records(), versions, commits))
    if len(curve) != training['optimizer_steps'] or initial == digest(model): raise ValueError('Fixed fresh200 M endpoint required')
    model.eval(); optimizer.zero_grad(set_to_none=True)
    return model, optimizer, {'optimizer_steps': len(curve), 'trainable_parameters': 161,
        'model_parameter_sha256_before': initial, 'model_parameter_sha256_after': digest(model),
        'episodes': summaries, 'loss_curve': curve, 'actual_optimizer_family_groups': dict(update_families),
        'actual_optimizer_target_bins': dict(target_bins), 'actual_optimizer_semantic_classes': len(classes),
        'actual_optimizer_policy_pseudo_classes': len(pseudo_classes),
        'GT_driven_memory_repair_or_deleted_wrong_history': False,
        'training_episode_resets_not_evaluator_GT_reset': True,
        'Novel_gain_official_CT_current_active_purity_self_exclusion_Pred_or_Goal_accepted': False}, tuple(private)
