#!/usr/bin/env python3
"""Actual Train-only predicted-memory D1/D2 rollouts, no oracle repair."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);args=p.parse_args()
    cp=ROOT/'configs/trackocd_core/persistent_policy_training.json';cfg=json.loads(cp.read_text())
    sources=('configs/trackocd_core/persistent_policy_training.json','scripts/trackocd_core/train_persistent_policy.py',
             'src/trackocd_core/persistent_policy.py','src/trackocd_core/train_first_replay.py','src/trackocd_core/train_first_experiment.py')
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=args.preregistration_commit:raise ValueError('Exact remote policy preregistration required')
    for n in sources:
        if subprocess.check_output(['git','show',f'{args.preregistration_commit}:{n}'])!=(ROOT/n).read_bytes():raise ValueError('Registered policy source changed')
    out=ROOT/'outputs/trackocd_core/core_training/train_first_v1/policy'
    if out.exists():raise ValueError('Preserve independent policy results; no overwrite/retry')
    import numpy as np
    import torch
    from torch.nn import functional as F
    from scripts.trackocd_core.run_gt_pilot_baselines import registered_orders
    from src.trackocd_core.train_first_experiment import load_train,frozen_model,evidence_bank,prototypes,evaluate
    from src.trackocd_core.train_first_replay import causal_rows,replay
    from src.trackocd_core.persistent_policy import CandidateMemory,DecisionMLP,SupervisionHistory,masked_logits,predicted_action
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);started=time.monotonic()
    cache,labels,known=load_train(ROOT);by={r['key']:r for r in labels}
    train=[r for r in cache.rows if by[r['key']]['partition']=='policy_train']
    dev=[r for r in cache.rows if by[r['key']]['partition']=='development']
    if len(train)!=229 or len(dev)!=258:raise ValueError('Registered full policy/development universe differs')
    category_counts=Counter(by[r['key']]['category_id'] for r in train)
    weights={c:len(train)/(len(category_counts)*n) for c,n in category_counts.items()}
    orders=registered_orders(sorted({r['video_id'] for r in train}));devorder=sorted({r['video_id'] for r in dev})
    thresholds={'known_cosine_threshold':.65,'existing_cosine_threshold':.55}
    out.mkdir(parents=True);fits=[];checkpoint_bytes=0

    def guard():
        mem={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable']<mem['MemTotal']*cfg['system_ram_reserve_fraction'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>cfg['host_planned_peak_bytes']:raise RuntimeError('Policy RAM guard')
        if time.monotonic()-started>cfg['training_wall_seconds']:raise RuntimeError('Policy wall guard')
        if checkpoint_bytes>cfg['frozen_checkpoint_payload_ceiling_bytes']:raise RuntimeError('Policy checkpoint budget')

    def development(model,bank,proto,bias=0.):
        cases=[]
        for cap in cfg['prefix_cycle']:
            sealed,runtime=replay(dev,devorder,cap,proto,thresholds,bank[cap],decision_model=model,wait_bias=bias)
            cases.append({'prefix':cap,**evaluate(sealed,dev,labels),'runtime':runtime})
        score=float(np.mean([.5*(c['standard']['h_score']+c['persistent']['correct_commit_ct']) for c in cases]))
        return {'partition':'development','score':score,'cases':cases,'final_heldout_opened':False}

    for representation in cfg['representation_families']:
        for seed in cfg['seeds']:
            frozen,lineage=frozen_model(ROOT,representation,seed)
            allroutes=train+dev;bank=evidence_bank(cache,allroutes,frozen);proto=prototypes(cache,labels,known,frozen)
            for name in cfg['models']:
                guard();torch.manual_seed(seed);model=DecisionMLP();initial=hashlib.sha256()
                for k,v in sorted(model.state_dict().items()):initial.update(k.encode());initial.update(v.numpy().tobytes())
                optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
                trace=[];checkpoints=[];input_digest=hashlib.sha256();fit_started=time.monotonic()
                for epoch in range(1,cfg['maximum_epochs']+1):
                    guard();epoch_stats=[]
                    for oi,(order_name,order) in enumerate(orders.items()):
                        cap=cfg['prefix_cycle'][(epoch-1+oi)%5];memory=CandidateMemory(proto);supervision=SupervisionHistory()
                        losses=[];probabilities=[];actions=[];cost_sum=0.;contaminated_later=0;nominal_hits=0;waits=0
                        for sequence,r in enumerate(causal_rows(train,order,cap)):
                            item=bank[cap][r['key']];label=by[r['key']]
                            input_digest.update(json.dumps((r['key'],cap,item['actual_prefix'])).encode())
                            features,k,t=memory.candidates(item['embedding'],item['uncertainty'],item['maturity'],item['quality'],item['elapsed'])
                            logits=masked_logits(model,torch.tensor(features));probability=torch.softmax(logits,dim=-1)
                            costs,target,credit=supervision.costs(label['category_id'],label['simulation_role']=='known',k,t,item['actual_prefix'],cfg['risk_constants'])
                            action=predicted_action(logits);prediction=memory.apply(action,k,t,item['embedding'],item['quality'])
                            nominal=F.cross_entropy(logits[None],torch.tensor([target]))
                            loss=nominal
                            if name=='D2_RISK_AWARE':
                                loss=loss+cfg['D2_expected_cost_weight']*(probability*torch.tensor(costs)).sum()
                                if credit is not None:
                                    # Credit a real earlier predicted write, retaining its graph.
                                    loss=loss+cfg['D2_delayed_credit_weight']*cfg['risk_constants']['later_predicted_memory_effect']*probabilities[credit][actions[credit]]
                                    contaminated_later+=1
                            losses.append(loss*weights[label['category_id']]);probabilities.append(probability);actions.append(action)
                            cost_sum+=float(costs[action]);nominal_hits+=action==target;waits+=action==3
                            supervision.update(prediction,label['category_id'],sequence)
                        loss=torch.stack(losses).mean()
                        if not torch.isfinite(loss):raise ValueError('Nonfinite policy loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm']);optimizer.step()
                        epoch_stats.append({'order':order_name,'prefix':cap,'loss':float(loss.detach()),'observations':len(train),
                            'predicted_action_cost_sum':cost_sum,'nominal_action_correct':nominal_hits,'waits':waits,
                            'persistent_states':len(memory.anonymous),'contaminated_states':sum(len(set(v))>1 for v in supervision.members.values()),
                            'delayed_pollution_penalties':contaminated_later,'memory_updated_by_predictions_only':True})
                    trace.append({'epoch':epoch,'episodes':epoch_stats})
                    if epoch in cfg['checkpoint_epochs']:
                        model.eval();dev_metrics=development(model,bank,proto);model.train()
                        path=out/f'{representation}_{name}_seed{seed}_epoch{epoch}.pt'
                        torch.save({'state_dict':model.state_dict(),'seed':seed,'model':name,'representation':representation,
                            'epoch':epoch,'policy_config_sha256':sha256_file(cp),'representation_lineage':lineage},path)
                        checkpoint_bytes+=path.stat().st_size
                        checkpoints.append({'epoch':epoch,'checkpoint':{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha256_file(path)},'development':dev_metrics})
                        atomic_json(out/'progress.json',{'stage':'REAL_PREDICTED_MEMORY_POLICY_TRAINING','representation':representation,
                            'model':name,'seed':seed,'epoch':epoch,'completed_fits':len(fits)})
                        print(json.dumps({'representation':representation,'model':name,'seed':seed,'epoch':epoch,'development_score':dev_metrics['score']}),flush=True)
                selected=max(checkpoints,key=lambda c:(c['development']['score'],-c['epoch']))
                saved=torch.load(ROOT/selected['checkpoint']['path'],weights_only=True,map_location='cpu');model.load_state_dict(saved['state_dict']);model.eval()
                curve=[{'wait_bias':bias,**development(model,bank,proto,bias)} for bias in cfg['wait_bias_grid']]
                operating={}
                for target in cfg['coverage_targets']:
                    def key(row):
                        coverage=float(np.mean([c['runtime']['commitment_coverage'] for c in row['cases']]))
                        error=float(np.mean([1-c['standard']['all_acc'] for c in row['cases']]))
                        return (abs(coverage-target),error,abs(row['wait_bias']))
                    chosen=min(curve,key=key);operating[str(target)]={'wait_bias':chosen['wait_bias'],
                        'actual_development_coverage':float(np.mean([c['runtime']['commitment_coverage'] for c in chosen['cases']])),
                        'target_coverage':target}
                fits.append({'representation':representation,'representation_lineage':lineage,'model':name,'seed':seed,
                    'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':initial.hexdigest(),
                    'training_inputs_order_prefix_sha256':input_digest.hexdigest(),'epochs':cfg['maximum_epochs'],'optimizer_steps':cfg['maximum_epochs']*len(orders),
                    'policy_train_tracks':len(train),'trace':trace,'checkpoints':checkpoints,'selected_checkpoint':selected,
                    'development_coverage_curve':curve,'operating_points':operating,'wall_seconds':time.monotonic()-fit_started})
                paired=[f for f in fits if f['representation']==representation and f['seed']==seed]
                if len(paired)==2:
                    assert len({f['initial_state_sha256'] for f in paired})==1
                    assert len({f['training_inputs_order_prefix_sha256'] for f in paired})==1
                atomic_json(out/'fits_partial.json',fits)
                del optimizer,model
            del frozen,bank
    receipt={'schema_version':'trackocd.core.main-policy-training.v1','status':'COMPLETE_REAL_TRAIN_PREDICTED_MEMORY_FITS_NOT_SCIENTIFIC_PASS',
        'preregistration_commit':args.preregistration_commit,'config_sha256':sha256_file(cp),'fits':fits,
        'source_sha256':{n:sha256_file(ROOT/n) for n in sources},'feature_manifest_sha256':sha256_file(ROOT/'outputs/trackocd_core/features/train_first_v1/manifest.json'),
        'GT_labels_used_only_for_targets_loss_evaluation':True,'GT_repaired_memory':False,'final_heldout_opened':False,
        'val_or_test_access':False,'DINO_detector_tracker_trained':False,'historical_outputs_overwritten':False,
        'resources':{'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_workers':1,
            'gpu_used':False,'checkpoint_bytes':checkpoint_bytes}}
    atomic_json(out/'training_receipt.json',receipt);atomic_json(ROOT/'outputs/trackocd_core/TRAIN_FIRST_POLICY_RESULT.json',receipt)
    print(json.dumps({'status':receipt['status'],'fits':len(fits),'resources':receipt['resources']}),flush=True)


if __name__=='__main__':main()
