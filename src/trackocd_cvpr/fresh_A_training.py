"""Fresh baseline optimizer; caller gates assets and owns resource guards.

Synthetic tests are plumbing only. Actual fit-only pilot must precede any
independent Fixed200 run; pilot weights are never a formal initialization.
"""
import hashlib
import json
import math
import torch
from .fresh_A_baseline import FAMILIES, Fit5BatchSampler, fresh_model, supervised_batch_loss
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest


def validate_protocol(cfg):
    if (cfg['families'] != list(FAMILIES) or cfg['training_seeds'] != [1027,1028,1029]
            or cfg['steps_per_family_and_seed'] != 200 or cfg['checkpoint_steps'] != [20,50,100,200]
            or cfg['optimizer'] != 'AdamW' or cfg['learning_rate'] != .0003
            or cfg['weight_decay'] != .0001 or cfg['gradient_clip_norm'] != 1.
            or cfg['geometry_weights'] != dict(zip(FAMILIES,(0.,1.,5.,0.,0.)))
            or cfg['pilot_ragged_numeric_qualification'] != {'every_actual_pilot_step':True,
                'embedding_atol':.000001,'embedding_rtol':.00001,
                'parameter_gradient_atol':.000001,'parameter_gradient_rtol':.0001,
                'interpretation':'Declared FP32 batched-versus-individual numeric/gradient qualification, not exact learned trajectory parity or scientific generalization'}
            or cfg['pilot'] != {'seed':1027,'steps':10,'families':list(FAMILIES),
                                'formal_warmstart':False,'development_labels_or_metrics':False}
            or cfg['batch'] != {'classes':5,'canonical_groups_per_class':2,'distinct_videos_per_class':2,
                'prefix_cycle':[1,2,4,8,16],'short_rule':'min(cap,observed), group actual lengths without padding',
                'same_seed_actual_batch_plan_all_families':True}
            or cfg['historical_old15_A1_A2_weights_labels_loader_or_selected_checkpoint_reuse']
            or cfg['policy_dev_final_Val_Test_targets_in_training_loss']
            or cfg['final_Val_Test_or_protected_input_access']):
        raise ValueError('Frozen fresh five-family/fit5/fixed endpoint protocol required')


def fit_baseline(tracks, cfg, family, seed, stage, guard, progress, save_checkpoint):
    validate_protocol(cfg)
    if family not in FAMILIES or seed not in (1027,1028,1029) or stage not in ('pilot','Fixed200'):
        raise ValueError('Explicit frozen family/seed/stage, no implicit expansion')
    if stage == 'pilot' and seed != 1027: raise ValueError('Pilot only fixed first seed')
    guard(); model=fresh_model(family,seed).train(); before=digest(model)
    adapter_before=digest(model.adapter); sampler=Fit5BatchSampler(tracks,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    steps=10 if stage=='pilot' else 200
    checkpoints=(10,) if stage=='pilot' else tuple(cfg['checkpoint_steps'])
    curve=[]; calendar=hashlib.sha256(); adapter_peak=0.; extra_peak=0.; seen_groups=set(); seen_videos=set()
    qualified_steps=0; embedding_error=0.; gradient_error=0.
    extra_name='reliability' if family=='A2_EVIDENCE' else 'static' if family=='A2_CAPACITY_CONTROL' else None
    for step in range(steps):
        guard(); plan=sampler.draw(step)
        # Private scheduling correspondence is only hashed, never a model input
        # or public per-track export. All families have identical complete plans.
        calendar.update(json.dumps([plan.fit_slots,plan.videos,plan.canonical_groups,
                                    plan.observed_counts],separators=(',',':')).encode())
        seen_groups.update(plan.canonical_groups); seen_videos.update(plan.videos)
        views=[t.visible(n) for t,n in zip(plan.tracks,plan.observed_counts)]
        optimizer.zero_grad(set_to_none=True)
        loss_args=(model,views,torch.tensor(plan.fit_slots),torch.tensor(plan.videos),torch.tensor(plan.canonical_groups))
        loss,embeddings=supervised_batch_loss(*loss_args)
        if not torch.isfinite(loss): raise ValueError('Finite actual loss, no silent skipped update')
        if stage=='pilot':
            guard(); reference,individual=supervised_batch_loss(*loss_args,individual_reference=True)
            tolerance=cfg['pilot_ragged_numeric_qualification']
            torch.testing.assert_close(embeddings,individual,
                atol=tolerance['embedding_atol'],rtol=tolerance['embedding_rtol'])
            embedding_error=max(embedding_error,float((embeddings-individual).abs().max().detach()))
            parameters=list(model.parameters())
            batched_grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            individual_grads=torch.autograd.grad(reference,parameters,allow_unused=True)
            for a,b in zip(batched_grads,individual_grads):
                if a is None or b is None:
                    if a is not b: raise ValueError('Reference gradient connectivity differs')
                    continue
                torch.testing.assert_close(a,b,atol=tolerance['parameter_gradient_atol'],
                    rtol=tolerance['parameter_gradient_rtol'])
                gradient_error=max(gradient_error,float((a-b).abs().max()))
            qualified_steps+=1
            del reference,individual,batched_grads,individual_grads; guard()
        loss.backward()
        adapter_norm=math.sqrt(sum(float(p.grad.detach().square().sum())
            for p in model.adapter.parameters() if p.grad is not None))
        extra_norm=0. if extra_name is None else math.sqrt(sum(float(p.grad.detach().square().sum())
            for p in getattr(model,extra_name).parameters() if p.grad is not None))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
        if not torch.isfinite(norm) or not math.isfinite(adapter_norm) or adapter_norm <= 0:
            raise ValueError('Actual finite nonzero adapter loss gradient, not weight-decay-only movement')
        if not math.isfinite(extra_norm): raise ValueError('Finite auxiliary loss gradient')
        adapter_peak=max(adapter_peak,adapter_norm); extra_peak=max(extra_peak,extra_norm)
        optimizer.step(); guard()
        row={'step':step+1,'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm),
             'adapter_loss_gradient_norm':adapter_norm,'extra_loss_gradient_norm':extra_norm,
             'visible_observations':sum(plan.observed_counts)}
        curve.append(row); progress(row,curve)
        if step+1 in checkpoints:
            save_checkpoint(model,optimizer,step+1,{'family':family,'seed':seed,'stage':stage,
                'batch_calendar_sha256':calendar.hexdigest(),'torch_rng_state':torch.get_rng_state()})
    after=digest(model)
    if before==after or torch.cuda.is_initialized(): raise ValueError('Actual fresh CPU update required')
    return model,optimizer,{'family':family,'seed':seed,'stage':stage,'completed_optimizer_steps':steps,
        'parameter_count':sum(p.numel() for p in model.parameters()),'scope':sampler.scope,
        'model_parameter_sha256_before':before,'model_parameter_sha256_after':after,
        'adapter_initialization_sha256':adapter_before,'batch_calendar_sha256':calendar.hexdigest(),
        'adapter_loss_gradient_norm_peak':adapter_peak,'extra_loss_gradient_norm_peak':extra_peak,
        'actually_sampled_canonical_groups':len(seen_groups),'actually_sampled_videos':len(seen_videos),
        'complete_eligible_census_is_not_claim_of_all_groups_visited':True,
        'batched_individual_numerically_qualified_steps':qualified_steps,
        'batched_individual_embedding_max_absolute_error':embedding_error,
        'batched_individual_parameter_gradient_max_absolute_error':gradient_error,
        'singleton_extra_gradient_may_be_zero':True,
        'parameter_movement_is_not_temporal_or_Novel_generalization_evidence':True,
        'loss_curve':curve,'dev_final_Val_Test_inputs_used':False,'training_GPU_hours':0}


def verify_matched_pilot(rows):
    if (len(rows)!=5 or [r['family'] for r in rows]!=list(FAMILIES)
            or any(r['stage']!='pilot' or r['seed']!=1027 or r['completed_optimizer_steps']!=10
                   or r['batched_individual_numerically_qualified_steps']!=10 for r in rows)
            or len({r['adapter_initialization_sha256'] for r in rows})!=1
            or len({r['batch_calendar_sha256'] for r in rows})!=1
            or any(r['scope']!=rows[0]['scope'] for r in rows)):
        raise ValueError('All five actual pilot runs, exact shared initialization/calendar/census')
    return {'all_five_pilot_families_completed':True,'actual_adapter_initialization_identical':True,
            'actual_private_batch_calendar_identical':True,'formal_warmstart':False,
            'formal15fits_or_development_or_Goal_accepted':False}


def checkpoint_state_equal(left,right):
    """Exact CPU tensor/optimizer structure reopen, not only model digest."""
    if type(left) is not type(right): return False
    if isinstance(left,torch.Tensor):
        return left.dtype==right.dtype and left.shape==right.shape and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(checkpoint_state_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return len(left)==len(right) and all(checkpoint_state_equal(a,b) for a,b in zip(left,right))
    return left==right
