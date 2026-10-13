"""Spread-calendar C exposure correction; caller gates real assets first.

Raw768/Top8/fresh708 in both arms. Independent prediction-owned histories;
common update frames require original Known-query loss in both arms. Raw Top8
Known proposals are static per current query, not anonymous-history/treatment
dependent; actual Known eligibility counts must agree before any update.
The repair adds .25 class-macro legal policy-pseudo Known-set rejection, never
an individual Existing/NEW/WAIT target. It is not posterior calibration.
No files, private targets or real experiments loaded at import or by this helper.
"""
from collections import Counter, defaultdict
import hashlib
import numpy as np
import torch
from .atomic_member_control import AtomicMemberControl
from .candidate_score_contract import CandidateScoreMLP
from .candidate_training_loss import (
    ExternalPredictionHistory, lawful_label, prediction_owned_loss_plan, candidate_supervised_loss)
from .candidate_scientific_training import class_macro_loss
from .complementary_known_loss import complementary_known_rejection
from .policy_training_data import policy_current_frames


def paired_candidate_fit(videos, projection, known, seed, *, episodes, updates_per_episode,
                         checkpoint_steps, guard, progress, checkpoint, repair_coefficient=.25, frozen_calendar=None):
    """A pair is NEW fresh training, not a historical control resume/replica.

    Zero repair coefficient is synthetic parity qualification only; a registered
    real caller must enforce fixed .25/three seeds/pilot-before-formal boundaries.
    Data scope/source/authority/live registration are the CALLER's responsibility.
    """
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    if (type(seed) is not int or seed not in (1027,1028,1029)
            or type(episodes) is not int or episodes < 1
            or type(updates_per_episode) is not int or updates_per_episode < 1
            or type(repair_coefficient) is not float or repair_coefficient not in (0., .25)
            or not videos or len({v for v,_ in videos}) != len(videos)):
        raise ValueError('Explicit independent seed/paired full episodes/fixed source coefficient')
    if frozen_calendar is None or len(frozen_calendar["selected_frames"]) != episodes or frozen_calendar["budget_each"] != updates_per_episode:
        raise ValueError("Actual fixed spread plan required before training")
    total_steps = episodes * updates_per_episode
    if (not checkpoint_steps or any(type(s) is not int or not 1 <= s <= total_steps for s in checkpoint_steps)
            or sorted(set(checkpoint_steps)) != list(checkpoint_steps) or checkpoint_steps[-1] != total_steps):
        raise ValueError('Explicit fixed checkpoints, no selected endpoint')
    guard()
    names = ('baseline', 'Known_set_rejection')
    models, optimizers, before = {}, {}, {}
    for name in names:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed); models[name] = CandidateScoreMLP().train()
        optimizers[name] = torch.optim.AdamW(models[name].parameters(),lr=.001,weight_decay=.0001)
        before[name] = digest(models[name])
    if before[names[0]] != before[names[1]]:
        raise ValueError('Actually identical fresh initializations')
    targets = {(v,str(t)): label for (v,t),label in projection.targets.items()}
    rng = np.random.default_rng(seed)
    curves, summaries, private = ({name: [] for name in names} for _ in range(3))
    calendar = hashlib.sha256(); shared_steps = 0
    for episode in range(episodes):
        order = rng.permutation(len(videos)).tolist()
        if order != frozen_calendar['orders'][episode]: raise ValueError('Same frozen video order')
        selected = set(frozen_calendar['selected_frames'][episode])
        if len(selected) != updates_per_episode: raise ValueError('All unique fixed update frames')
        pending = {name: [] for name in names}; controls, histories = {}, {}
        steps = 0
        for name in names:
            def scorer(features, context, arm=name):
                with torch.set_grad_enabled(steps < updates_per_episode):
                    logits = models[arm](torch.tensor(features.copy()),torch.tensor(context.copy()))
                pending[arm].append(logits)
                return logits.detach().numpy().copy()
            controls[name] = AtomicMemberControl(known,scorer,K=8)
            histories[name] = ExternalPredictionHistory()
        counts = {name: Counter() for name in names}
        actions = {name: Counter() for name in names}
        sealed = {name: [] for name in names}
        frames = observed = empty = 0
        for position, ordinal in enumerate(order):
            video, arrays = videos[ordinal]
            if position:
                for control in controls.values(): control.start_next_video()
            for frame, rows in policy_current_frames(arrays):
                guard()
                result, base, rejection, families = {}, {}, {}, {}
                for name in names:
                    pending[name].clear(); prior = histories[name].snapshot()
                    result[name] = controls[name].ingest(frame,rows)
                    if len(pending[name]) != len(result[name].events):
                        raise ValueError('Actual pre-update numeric scorer/event alignment')
                    base[name], rejection[name], families[name] = defaultdict(list), defaultdict(list), Counter()
                    for logits, aliases, event in zip(pending[name],result[name].routing_groups,result[name].events):
                        if not np.array_equal(logits.detach().numpy(),np.frombuffer(event.score_bytes,dtype='<f4')):
                            raise ValueError('Actual prediction and external loss use same pre-update weights')
                        keys = [(video,route) for route in aliases]
                        plan = prediction_owned_loss_plan(event.candidate_bindings_for_external_loss,keys,targets,prior)
                        original_loss = candidate_supervised_loss(logits,plan)
                        extra = complementary_known_rejection(logits,event.candidate_bindings_for_external_loss,plan.query_family)
                        label = lawful_label(targets.get(keys[0],('no_retained_semantic_truth',None)))
                        if original_loss is None: counts[name]['base_masked_groups_all_frames'] += 1
                        else:
                            base[name][label].append(original_loss); families[name][plan.query_family] += 1
                        if extra is not None:
                            if label is None or label[0] != 'policy_pseudo':
                                raise ValueError('No invented Novel/unknown/individual-action supervision')
                            rejection[name][label].append(extra)
                            counts[name]['rejection_eligible_groups_all_frames'] += 1
                        actions[name][event.action] += len(aliases)
                if result[names[0]].routing_groups != result[names[1]].routing_groups:
                    raise ValueError('Same actual physical Raw alias groups in both arms')
                if families[names[0]]['Known'] != families[names[1]]['Known']:
                    raise ValueError('Known-anchor calendar cannot depend on treatment/anonymous history')
                eligible = families[names[0]]['Known'] > 0
                planned = (episode,position,frame) in selected
                if planned and not eligible: raise ValueError("Frozen Known anchor unavailable; no substitute frame")
                if planned and steps < updates_per_episode:
                    # Hash administrative frame/ordinal only for PRIVATE pairing
                    # correspondence, never forward input or public row export.
                    calendar.update(np.asarray([episode,position,frame],dtype='<i8').tobytes())
                    steps += 1; shared_steps += 1
                    for name in names:
                        base_loss = class_macro_loss(base[name])
                        reject_loss = class_macro_loss(rejection[name]) if rejection[name] else None
                        coefficient = repair_coefficient if name == 'Known_set_rejection' else 0.
                        extra_norm = 0.
                        if coefficient and reject_loss is not None:
                            gradients = torch.autograd.grad(coefficient*reject_loss,
                                tuple(models[name].parameters()),retain_graph=True,allow_unused=True)
                            extra_norm = sum(float(g.square().sum()) for g in gradients if g is not None)**.5
                            del gradients
                        loss = base_loss if not coefficient or reject_loss is None else base_loss+coefficient*reject_loss
                        optimizer = optimizers[name]; optimizer.zero_grad(set_to_none=True)
                        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(models[name].parameters(),1.)
                        if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0 or not np.isfinite(extra_norm):
                            raise ValueError('Actual finite nonzero loss gradient, not a decay-only update')
                        optimizer.step()
                        counts[name]['available_rejection_groups_on_updated_frames'] += sum(map(len,rejection[name].values()))
                        counts[name]['updates_with_rejection_available'] += int(reject_loss is not None)
                        counts[name]['updates_with_rejection_loss_gradient'] += int(extra_norm > 0)
                        row = {'step': shared_steps,'episode':episode,'episode_optimizer_step':steps,
                            'loss':float(loss.detach()),'base_loss':float(base_loss.detach()),
                            'rejection_macro_loss':None if reject_loss is None else float(reject_loss.detach()),
                            'rejection_coefficient':coefficient,'rejection_parameter_gradient_norm':extra_norm,
                            'gradient_norm_before_clip':float(norm),'base_family_groups':dict(families[name]),
                            'base_supervised_classes':len(base[name]),'rejection_supervised_classes':len(rejection[name]),
                            'shared_physical_calendar_sha256':calendar.hexdigest()}
                        curves[name].append(row)
                        if shared_steps in checkpoint_steps:
                            checkpoint(name,models[name],optimizer,shared_steps,episode,steps,rng.bit_generator.state)
                    progress({name:curves[name][-1] for name in names},curves)
                for name in names:
                    histories[name].observe_frame(video,result[name],targets)
                    sealed[name].append((video,result[name]))
                frames += 1; empty += int(not rows); observed += len(rows); guard()
        if steps != updates_per_episode:
            raise ValueError('Whole shared update budget unmet, no order/frame replacement or partial acceptance')
        for name in names:
            bank,next_id = controls[name].state_snapshot()
            summaries[name].append({'episode':episode,'optimizer_steps':steps,'videos':len(videos),
                'metadata_frames':frames,'empty_frames':empty,'physical_observations':observed,
                'physical_actions':dict(actions[name]),'loss_opportunities':dict(counts[name]),
                'actual_bank_commits_retained':len(bank.commit_history()),
                'actual_anonymous_commits_retained':len(histories[name].records()),'allocated_tokens':next_id})
            private[name].append((order,tuple(sealed[name]),histories[name].records()))
    if calendar.hexdigest() != frozen_calendar["selected_calendar_sha256"]:
        raise ValueError("Actually executed entire frozen calendar")
    after = {name:digest(model) for name,model in models.items()}
    if shared_steps != total_steps or any(before[name] == after[name] for name in names):
        raise ValueError('Both actual fresh full-budget fits required')
    for model in models.values(): model.eval()
    for optimizer in optimizers.values(): optimizer.zero_grad(set_to_none=True)
    private = {name:tuple(rows) for name,rows in private.items()}
    return models,optimizers,{
        'arms':names,'optimizer_steps_each':shared_steps,'trainable_parameters_each':708,
        'initialization_sha256':before,'endpoint_parameter_sha256':after,
        'actual_shared_physical_calendar_sha256':calendar.hexdigest(),
        'offline_Known_uniform_rank_plan_rule':frozen_calendar['rule'],
        'offline_eligible_Known_frames_each':frozen_calendar['eligible_Known_frame_counts'],
        'frozen_plan_used_before_updates_not_model_input':True,
        'episodes':summaries,'loss_curves':curves,
        'same_update_frames_not_identical_endogenous_candidate_or_loss_opportunities':True,
        'GT_repair_unknown_as_background_or_individual_pseudo_action_target_added':False,
        'calibration_development_Pred_joint_or_Goal_accepted':False,
    },private
