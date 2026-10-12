"""Fixed multiepisode C fits with full endogenous prediction-owned histories.

Raw-only isolates C from any selected R checkpoint. Training episodes reset
their own simulation bank, never a live evaluation lineage or GT-triggered
repair. All episode mistakes stay in private immutable traces. Matched nominal
capacity/input/steps does not imply matched function class, active information
or identical endogenous loss availability. Not a qualified Pred or M trial.
"""
from collections import Counter, defaultdict
import numpy as np
import torch

from .atomic_member_control import AtomicMemberControl
from .candidate_score_contract import CandidateScoreMLP
from .candidate_training_loss import (
    ExternalPredictionHistory, lawful_label, prediction_owned_loss_plan, candidate_supervised_loss,
)
from .policy_training_data import policy_current_frames


def variant_numeric_features(features, variant):
    """Pool-control columns2:5 are SHARED pool context, not Known support stats.

    Same 708 parameters, distinct information/function classes explicitly
    acknowledged. Row permutation and output IDs cannot alter pooled inputs.
    Mean reduction uses canonical numeric order, not candidate/token IDs.
    """
    result = np.array(features, dtype=np.float32, copy=True)
    if variant == 'Top8_pool_context' and len(result):
        pooled = np.sort(result[:, 2:5], axis=0).astype(np.float64).mean(axis=0).astype(np.float32)
        result[:, 2:5] = pooled
    elif variant not in ('Top1_conditioned', 'Top8_conditioned', 'Top8_pool_context'):
        raise ValueError('Only fixed candidate-direction controls')
    return result


def class_macro_loss(group_losses):
    # Numeric canonical reductions prevent external class-ID/order changes
    # becoming floating accumulation-order differences. Labels are LOSS only.
    per_class = [torch.stack(terms).sort().values.mean() for terms in group_losses.values()]
    return torch.stack(per_class).sort().values.mean()


def scientific_candidate_fit(videos, projection, known_prototypes, window, variant, seed,
                             guard, progress, checkpoint):
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    settings = window['variants'][variant]; training = window['training']
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = CandidateScoreMLP().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    initial_digest = digest(model)
    targets = {(v, str(t)): label for (v, t), label in projection.targets.items()}
    curve = []; episode_summaries = []; private_episodes = []
    update_families = Counter(); update_classes = set(); update_pseudo_classes = set()
    fixed = training['updates_per_episode']
    for episode in range(training['episodes']):
        order = rng.permutation(len(videos)).tolist(); pending = []; steps = 0
        def score(features, context):
            numeric = variant_numeric_features(features, variant)
            with torch.set_grad_enabled(steps < fixed):
                logits = model(torch.tensor(numeric), torch.tensor(context.copy()))
            pending.append(logits)
            return logits.detach().numpy().copy()
        control = AtomicMemberControl(known_prototypes, score, K=settings['K_per_family'])
        history = ExternalPredictionHistory()
        counts = Counter(); actions = Counter(); all_families = Counter()
        frames = observations = empty = groups = 0; sealed = []
        independent_positive_groups = cross_video_positive_groups = 0
        for position, ordinal in enumerate(order):
            video, arrays = videos[ordinal]
            if position: control.start_next_video()
            for frame, rows in policy_current_frames(arrays):
                guard(); pending.clear(); prior = history.snapshot()
                result = control.ingest(frame, rows)
                if len(pending) != len(result.events): raise ValueError('Actual atomic scorer/event alignment')
                terms = defaultdict(list); frame_families = Counter(); masked = 0
                other = cross = 0
                for logits, aliases, event in zip(pending, result.routing_groups, result.events):
                    if not np.array_equal(logits.detach().numpy(), np.frombuffer(event.score_bytes, dtype='<f4')):
                        raise ValueError('Actual rollout and loss logits must have same weights')
                    keys = [(video, route) for route in aliases]
                    plan = prediction_owned_loss_plan(event.candidate_bindings_for_external_loss, keys, targets, prior)
                    loss = candidate_supervised_loss(logits, plan); counts[plan.reason] += 1
                    if loss is None:
                        masked += 1
                    else:
                        label = lawful_label(targets[keys[0]])
                        terms[label].append(loss); frame_families[plan.query_family] += 1
                        other += int(plan.positive_has_other_physical_member)
                        cross += int(plan.positive_has_other_video_member)
                    actions[event.action] += len(aliases)
                all_families.update(frame_families)
                independent_positive_groups += other; cross_video_positive_groups += cross
                if terms and steps < fixed:
                    loss = class_macro_loss(terms)
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'])
                    if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
                        raise ValueError('Finite actual C optimizer gradient required')
                    optimizer.step(); steps += 1
                    update_families.update(frame_families); update_classes.update(terms)
                    update_pseudo_classes.update(label for label in terms if label[0] == 'policy_pseudo')
                    row = {'step': len(curve) + 1, 'episode': episode, 'episode_optimizer_step': steps,
                        'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
                        'actual_optimizer_family_groups': dict(frame_families),
                        'supervised_classes_this_update': len(terms), 'masked_groups_this_frame': masked,
                        'positive_other_physical_groups_this_update': other,
                        'positive_other_video_groups_this_update': cross}
                    curve.append(row); progress(row, curve)
                    if len(curve) in training['checkpoint_steps']:
                        checkpoint(model, optimizer, len(curve), episode, steps, rng.bit_generator.state)
                history.observe_frame(video, result, targets)
                sealed.append((video, result)); frames += 1; empty += int(not rows)
                observations += len(rows); groups += len(result.events); guard()
        if steps != fixed: raise ValueError('Fixed optimizer count unmet, no successful subset or alternate order')
        bank, next_id = control.state_snapshot()
        episode_summaries.append({'episode': episode, 'optimizer_steps': steps,
            'videos': len(videos), 'metadata_frames': frames, 'empty_frames': empty,
            'physical_observations': observations, 'canonical_groups': groups,
            'physical_actions': dict(actions), 'loss_plan_counts_all_frames': dict(counts),
            'eligible_family_groups_all_frames_including_frozen_tail': dict(all_families),
            'positive_other_physical_groups_all_frames': independent_positive_groups,
            'positive_other_video_groups_all_frames': cross_video_positive_groups,
            'actual_anonymous_commits_retained': len(history.records()),
            'actual_bank_commits_retained': len(bank.commit_history()), 'allocated_tokens': next_id})
        private_episodes.append((order, tuple(sealed), history.records()))
    if len(curve) != training['optimizer_steps'] or all(torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise ValueError('Fixed fresh C fit required')
    model.eval(); optimizer.zero_grad(set_to_none=True)
    return model, optimizer, {
        'optimizer_steps': len(curve), 'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'model_parameter_sha256_before': initial_digest,
        'episodes': episode_summaries, 'loss_curve': curve,
        'actual_optimizer_family_groups': dict(update_families), 'actual_optimizer_semantic_classes': len(update_classes),
        'actual_optimizer_policy_pseudo_classes': len(update_pseudo_classes),
        'both_Known_and_pseudo_received_optimizer_loss': bool(update_families['Known'] and update_families['policy_pseudo']),
        'GT_repair_or_deleted_wrong_history': False,
        'training_episode_resets_not_evaluation_GT_reset': True,
        'self_support_exclusion_or_Pred_or_M_or_Goal_acceptance': False,
    }, tuple(private_episodes)
