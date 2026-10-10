#!/usr/bin/env python3
"""Frozen D1/D2/ablations, all orders/caps, no heldout operating-point tuning."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,atomic_write_text,sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);args=p.parse_args()
    cp=ROOT/'configs/trackocd_core/gt_main_evaluation.json';cfg=json.loads(cp.read_text())
    sources=('configs/trackocd_core/gt_main_evaluation.json','scripts/trackocd_core/evaluate_persistent_policy.py',
        'src/trackocd_core/train_first_experiment.py','src/trackocd_core/train_first_replay.py','src/trackocd_core/scientific_statistics.py')
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=args.preregistration_commit:raise ValueError('Exact remote policy evaluation preregistration required')
    for n in sources:
        if subprocess.check_output(['git','show',f'{args.preregistration_commit}:{n}'])!=(ROOT/n).read_bytes():raise ValueError('Policy evaluation source changed')
    out=ROOT/'outputs/trackocd_core/core_training/train_first_v1/policy_evaluation'
    if out.exists():raise ValueError('Preserve completed heldout; no hidden redesign/retry')
    import numpy as np
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.trackocd_core.persistent_policy import DecisionMLP
    from src.trackocd_core.train_first_experiment import load_train,frozen_model,evidence_bank,prototypes,evaluate
    from src.trackocd_core.train_first_replay import replay
    from src.trackocd_core.scientific_statistics import paired_video_bootstrap
    from scripts.trackocd_core.run_gt_pilot_baselines import registered_orders
    torch.set_num_threads(1);started=time.monotonic()
    policy_path=ROOT/'outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json'
    training=json.loads(policy_path.read_text())
    if training['config_sha256']!=sha256_file(ROOT/'configs/trackocd_core/persistent_policy_training.json'):raise ValueError('Policy config differs')
    for n,sha in training['source_sha256'].items():
        if sha256_file(ROOT/n)!=sha:raise ValueError('Policy training source changed')
    cache,labels,known=load_train(ROOT);by={r['key']:r for r in labels}
    routes=[r for r in cache.rows if by[r['key']]['partition']=='final_heldout_selection']
    orders=registered_orders(sorted({r['video_id'] for r in routes}));thresholds={}
    # Same representation per risk-only pair; each ablation explicit, not mislabeled.
    variants=[('D1_SIMPLE_MLP','A1_SELECTED','D1_SIMPLE_MLP',{}),('D2_RISK_AWARE','A1_SELECTED','D2_RISK_AWARE',{}),
        ('WITHOUT_ADAPTER','A0_RAW','D2_RISK_AWARE',{}),('WITHOUT_TEMPORAL','A1_SELECTED','D2_RISK_AWARE',{}),
        ('WITHOUT_RELIABILITY','A2_EVIDENCE','D2_RISK_AWARE',{'no_reliability':True}),
        ('WITHOUT_WAIT','A2_EVIDENCE','D2_RISK_AWARE',{'allow_wait':False}),
        ('WITHOUT_MEMORY','A2_EVIDENCE','D2_RISK_AWARE',{'no_memory':True}),
        ('WITHOUT_RISK','A2_EVIDENCE','D1_SIMPLE_MLP',{}),
        ('RESET_PER_VIDEO','A2_EVIDENCE','D2_RISK_AWARE',{'reset_per_video':True}),('FULL','A2_EVIDENCE','D2_RISK_AWARE',{})]
    out.mkdir(parents=True);cases=[];ledgers=[];modules=[]
    for seed in cfg['seeds']:
        for name,representation,policy,options in variants:
            model,lineage=frozen_model(ROOT,representation,seed);bank=evidence_bank(cache,routes,model,options.get('no_reliability',False))
            if options.get('no_reliability',False):
                proto_rows=[r for r in cache.rows if by[r['key']]['partition']=='prototype' and by[r['key']]['category_id'] in known]
                protobank=evidence_bank(cache,proto_rows,model,True)[16];grouped=defaultdict(list)
                for r in proto_rows:grouped[by[r['key']]['category_id']].append(protobank[r['key']]['embedding'])
                proto={c:np.mean(values,axis=0) for c,values in grouped.items()}
                proto={c:v/max(np.linalg.norm(v),1e-12) for c,v in proto.items()}
                del protobank
            else:proto=prototypes(cache,labels,known,model)
            fit=next(f for f in training['fits'] if (f['seed'],f['representation'],f['model'])==(seed,representation,policy))
            checkpoint=fit['selected_checkpoint']['checkpoint'];path=ROOT/checkpoint['path']
            if path.stat().st_size!=checkpoint['bytes'] or sha256_file(path)!=checkpoint['sha256']:raise ValueError('Frozen policy checkpoint differs')
            saved=torch.load(path,weights_only=True,map_location='cpu');decision=DecisionMLP().eval().requires_grad_(False);decision.load_state_dict(saved['state_dict'],strict=True)
            modules.append({'method':name,'seed':seed,'representation':representation,'policy':policy,'options':options,
                'policy_checkpoint':checkpoint,'representation_lineage':lineage,'parameters':708,'alias_D2_A1':name=='WITHOUT_TEMPORAL'})
            for target,point in fit['operating_points'].items():
                for order_name,order in orders.items():
                    for cap in cfg['prefixes']:
                        kwargs={k:v for k,v in options.items() if k!='no_reliability'}
                        sealed,runtime=replay(routes,order,cap,proto,thresholds,bank[cap],decision_model=decision,wait_bias=point['wait_bias'],**kwargs)
                        metrics=evaluate(sealed,routes,labels)
                        cases.append({'method':name,'seed':seed,'representation':representation,'coverage_target':float(target),
                            'development_operating_point':point,'order':order_name,'prefix':cap,**metrics,'runtime':runtime})
                        if float(target)==1.0:
                            for e in sealed.events:
                                ledgers.append({'method':name,'seed':seed,'order':order_name,'prefix':cap,'sequence':e.sequence,'video_id':e.physical_key.video_id,
                                    'physical_track_id':e.physical_key.local_track_id,'kind':e.kind,'known_category_id':e.known_category_id,'token':e.token})
            atomic_json(out/'progress.json',{'method':name,'seed':seed,'actual_cases':len(cases)});print(json.dumps({'method':name,'seed':seed,'cases':len(cases)}),flush=True)
            del bank,model,decision
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>cfg['host_planned_peak_bytes'] or time.monotonic()-started>cfg['wall_seconds']:raise RuntimeError('Policy heldout resource guard')
    aggregate=[];metrics={'old_acc':'standard','new_acc':'standard','h_score':'standard','all_acc':'standard','correct_commit_ct':'persistent',
        'false_merge_rate':'persistent','false_split_new_rate':'persistent','wrong_known_assignment_rate':'persistent','wait_unresolved_rate':'persistent',
        'effective_commit_coverage':'persistent','all_novel_wrong_known_rate':'errors','memory_contamination_write_events':'errors'}
    for name,representation,policy,options in variants:
        for target in (.5,.75,1.):
            for cap in cfg['prefixes']:
                rows=[r for r in cases if (r['method'],r['coverage_target'],r['prefix'])==(name,target,cap)]
                row={'scope':'TAO_TRAIN_GT_CONTROLLED_HELDOUT','method':name,'representation':representation,'coverage_target':target,'prefix':cap,'seeds':3,'orders':4}
                for metric,section in metrics.items():
                    means=[float(np.mean([r[section][metric] for r in rows if r['seed']==s])) for s in cfg['seeds']]
                    row[metric+'_mean']=float(np.mean(means));row[metric+'_seed_std_ddof0']=float(np.std(means))
                row['actual_commitment_coverage_mean']=float(np.mean([r['runtime']['commitment_coverage'] for r in rows]))
                aggregate.append(row)
    paired=[]
    for left,right in (('D1_SIMPLE_MLP','D2_RISK_AWARE'),('WITHOUT_RISK','FULL'),('WITHOUT_TEMPORAL','FULL')):
        for target in (.5,.75,1.):
            for seed in cfg['seeds']:
                for order in orders:
                    for cap in cfg['prefixes']:
                        a,b=[next(r for r in cases if (r['method'],r['coverage_target'],r['seed'],r['order'],r['prefix'])==(n,target,seed,order,cap)) for n in (left,right)]
                        gap=abs(a['persistent']['effective_commit_coverage']-b['persistent']['effective_commit_coverage'])
                        total_gap=abs(a['runtime']['commitment_coverage']-b['runtime']['commitment_coverage'])
                        record={'left':left,'right':right,'coverage_target':target,'seed':seed,'order':order,'prefix':cap,
                            'actual_reuse_opportunity_coverage_gap':gap,'actual_total_commitment_coverage_gap':total_gap,
                            'comparable_within_registered_tolerance':max(gap,total_gap)<=.05,
                            'CT_delta':b['persistent']['correct_commit_ct']-a['persistent']['correct_commit_ct'],
                            'false_merge_delta':b['persistent']['false_merge_rate']-a['persistent']['false_merge_rate'],
                            'all_novel_wrong_known_delta':b['errors']['all_novel_wrong_known_rate']-a['errors']['all_novel_wrong_known_rate']}
                        if target==1. and cap==16:
                            record['conditional_video_bootstrap']=paired_video_bootstrap(a['errors']['per_video_conditional_fixed_mapping_counts'],b['errors']['per_video_conditional_fixed_mapping_counts'],cfg['paired_bootstrap']['samples'],cfg['paired_bootstrap']['seed'])
                        paired.append(record)
    path=out/'sealed_predictions.parquet';pq.write_table(pa.Table.from_pylist(ledgers),path)
    receipt={'schema_version':'trackocd.core.main-policy-heldout.v1','status':'COMPLETE_REAL_TRAIN_HELDOUT_POLICY_NOT_PREDICTED_RESULT',
        'preregistration_commit':args.preregistration_commit,'config_sha256':sha256_file(cp),'training_receipt_sha256':sha256_file(policy_path),
        'cases':cases,'aggregate':aggregate,'modules':modules,'paired_comparisons':paired,'known_gt':207,'pseudo_novel_gt':16,'orders':orders,
        'all_seeds_orders_prefixes_and_coverage_points_kept':True,'heldout_operating_point_tuning':False,'GT_or_mapping_in_policy_state':False,
        'val_or_test_access':False,'source_sha256':{n:sha256_file(ROOT/n) for n in sources},
        'private_ledger':{'bytes':path.stat().st_size,'sha256':sha256_file(path),'rows':len(ledgers)},
        'resources':{'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_workers':1,'gpu_used':False}}
    atomic_json(out/'results.json',receipt);atomic_json(ROOT/'outputs/trackocd_core/TRAIN_FIRST_POLICY_HELDOUT_RESULT.json',receipt)
    buffer=io.StringIO();w=csv.DictWriter(buffer,fieldnames=list(aggregate[0]),lineterminator='\n');w.writeheader();w.writerows(aggregate)
    atomic_write_text(ROOT/'outputs/trackocd_core/PERSISTENT_POLICY_COMPARISON.csv',buffer.getvalue())
    print(json.dumps({'status':receipt['status'],'cases':len(cases),'resources':receipt['resources'],'heldout_p16_maxcoverage':[r for r in aggregate if r['prefix']==16 and r['coverage_target']==1.]}),flush=True)


if __name__=='__main__':main()
