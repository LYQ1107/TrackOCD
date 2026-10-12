"""Ten-step C optimizer plumbing on a complete prediction-owned GT rollout.

Raw evidence only, no R checkpoint choice. Not a scientific C/M direction
acceptance, jointly trained R/C/M, self-excluded reuse or a Pred benchmark.
After the fixed tenth update the scorer freezes but ALL remaining frames and
masked physical observations still run. GT supplies external loss only.
"""
from collections import Counter
import numpy as np
import torch

from .atomic_member_control import AtomicMemberControl
from .candidate_score_contract import CandidateScoreMLP
from .candidate_training_loss import ExternalPredictionHistory, prediction_owned_loss_plan, candidate_supervised_loss
from .policy_training_data import policy_current_frames


def fit_candidate_pilot(videos, projection, known_prototypes, cfg, guard, progress):
    torch.manual_seed(cfg['seed'])
    model = CandidateScoreMLP().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    pending = []; updates = 0
    def score(features, context):
        # Only numeric features/context, no bindings, roles, IDs or GT.
        with torch.set_grad_enabled(updates < cfg['optimizer_steps']):
            logits = model(torch.tensor(features.copy()), torch.tensor(context.copy()))
        pending.append(logits)
        return logits.detach().numpy().copy()
    control = AtomicMemberControl(known_prototypes, score, K=cfg['K_per_family'])
    history = ExternalPredictionHistory()
    # Binding re-key is purely external; never passed to numeric forward/bank.
    targets = {(v, str(t)): label for (v, t), label in projection.targets.items()}
    plans_count = Counter(); family_count = Counter(); actions = Counter()
    frames = empty = observations = groups = 0; curve = []; sealed = []
    for ordinal, (video, arrays) in enumerate(videos):
        if ordinal: control.start_next_video()
        for frame, rows in policy_current_frames(arrays):
            guard(); pending.clear(); prior = history.snapshot()
            result = control.ingest(frame, rows)  # Own predictions commit BEFORE GT loss.
            if len(pending) != len(result.events): raise ValueError('One numeric score per canonical physical group')
            terms = []
            for logits, aliases, event in zip(pending, result.routing_groups, result.events):
                if not np.array_equal(logits.detach().numpy(), np.frombuffer(event.score_bytes, dtype='<f4')):
                    raise ValueError('Loss and actual rollout used different scorer weights/logits')
                plan = prediction_owned_loss_plan(event.candidate_bindings_for_external_loss,
                    [(video, route) for route in aliases], targets, prior)
                plans_count[plan.reason] += 1
                loss = candidate_supervised_loss(logits, plan)
                if loss is not None:
                    terms.append(loss); family_count[plan.query_family] += 1
                actions[event.action] += len(aliases)
            if terms and updates < cfg['optimizer_steps']:
                loss = torch.stack(terms).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
                    raise ValueError('Real finite C gradient required')
                optimizer.step(); updates += 1
                curve.append({'step': updates, 'loss': float(loss.detach()),
                    'gradient_norm_before_clip': float(norm), 'supervised_groups_this_frame': len(terms)})
                progress(curve[-1])
            history.observe_frame(video, result, targets)  # Appends actual outputs, never edits controller.
            sealed.append(result); frames += 1; empty += int(not rows)
            observations += len(rows); groups += len(result.events)
            guard()
    if updates != cfg['optimizer_steps'] or all(torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise ValueError('Fixed actual optimizer endpoint, no successful zero-fit subset')
    model.eval(); optimizer.zero_grad(set_to_none=True)
    bank, next_id = control.state_snapshot()
    summary = {'optimizer_steps': updates, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'complete_rollout': {'videos': len(videos), 'metadata_frames': frames, 'empty_frames': empty,
            'physical_observations': observations, 'canonical_groups': groups, 'physical_actions': dict(actions)},
        'loss_plan_counts_all_frames': dict(plans_count), 'eligible_loss_family_counts_all_frames': dict(family_count),
        'loss_curve': curve, 'actual_anonymous_commits_retained': len(history.records()),
        'actual_bank_commits_retained': len(bank.commit_history()), 'allocated_tokens': next_id,
        'all_remaining_frames_after_fixed_endpoint_run_without_optimizer': True,
        'GT_created_reassigned_or_repaired_bank_or_removed_wrong_history': False,
        'self_support_exclusion_or_independent_category_reuse_or_C_M_scientific_acceptance': False}
    return model, optimizer, summary, tuple(sealed), history.records()
