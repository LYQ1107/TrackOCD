#!/usr/bin/env python3
"""First full three-seed/four-order/five-prefix Train heldout result table."""
from __future__ import annotations
import argparse
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


def thresholds(known,existing):
    return {'known_cosine_threshold':known,'existing_cosine_threshold':existing,
        'dpmeans_known_distance_threshold':1-known,'dpmeans_lambda_distance':1-existing}


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);args=p.parse_args()
    cp=ROOT/'configs/trackocd_core/gt_main_evaluation.json';cfg=json.loads(cp.read_text())
    sources=('configs/trackocd_core/gt_main_evaluation.json','scripts/trackocd_core/evaluate_core_representation.py',
        'src/trackocd_core/train_first_experiment.py','src/trackocd_core/train_first_replay.py')
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=args.preregistration_commit:raise ValueError('Exact remote heldout preregistration required')
    for n in sources:
        if subprocess.check_output(['git','show',f'{args.preregistration_commit}:{n}'])!=(ROOT/n).read_bytes():raise ValueError('Heldout source changed')
    out=ROOT/'outputs/trackocd_core/core_training/train_first_v1/representation_evaluation'
    if out.exists():raise ValueError('Preserve final heldout results; no redesign/retry')
    import numpy as np
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.trackocd_core.run_gt_pilot_baselines import registered_orders
    from src.trackocd_core.train_first_experiment import load_train,frozen_model,evidence_bank,prototypes,evaluate,diagnostics
    from src.trackocd_core.train_first_replay import replay
    torch.set_num_threads(1);started=time.monotonic();cache,labels,known=load_train(ROOT);by={r['key']:r for r in labels}
    dev=[r for r in cache.rows if by[r['key']]['partition']=='development']
    heldout=[r for r in cache.rows if by[r['key']]['partition']==cfg['heldout_partition']]
    if len(dev)!=258 or len(heldout)!=223:raise ValueError('Full registered selection universe required')
    orders=registered_orders(sorted({r['video_id'] for r in heldout}));devorder=sorted({r['video_id'] for r in dev})
    out.mkdir(parents=True);cases=[];calibrations=[];ledgers=[];modules=[]
    for representation in cfg['representation_variants']:
        seeds=[None] if representation=='A0_RAW' else cfg['seeds']
        for seed in seeds:
            model,lineage=frozen_model(ROOT,representation,seed);proto=prototypes(cache,labels,known,model)
            # Development frozen operating point chosen before heldout vectors/results.
            devbank=evidence_bank(cache,dev,model)
            names=cfg['baselines_raw'] if model is None else [cfg['learned_representation_backend']]
            chosen={}
            for backend in names:
                trials=[]
                for k in cfg['known_threshold_grid']:
                    for e in cfg['existing_threshold_grid']:
                        th=thresholds(k,e);sealed,runtime=replay(dev,devorder,16,proto,th,devbank[16],name=backend)
                        metrics=evaluate(sealed,dev,labels);score=.5*(metrics['standard']['h_score']+metrics['persistent']['correct_commit_ct'])
                        trials.append({'known':k,'existing':e,'thresholds':th,'score':score,'development':metrics})
                selected=max(trials,key=lambda t:(t['score'],-t['known'],-t['existing']))
                chosen[backend]=selected['thresholds'];calibrations.append({'representation':representation,'seed':seed,'backend':backend,
                    'partition':'development','selected':selected,'all25trials':trials,'heldout_used':False})
            del devbank
            bank=evidence_bank(cache,heldout,model)
            modules.append({'representation':representation,'seed':seed,'lineage':lineage,'parameters':0 if model is None else sum(p.numel() for p in model.parameters())})
            diags={cap:diagnostics(bank,heldout,labels,proto,cap) for cap in cfg['prefixes']}
            for backend in names:
                for order_name,order in orders.items():
                    for cap in cfg['prefixes']:
                        sealed,runtime=replay(heldout,order,cap,proto,chosen[backend],bank[cap],name=backend)
                        metrics=evaluate(sealed,heldout,labels)
                        cases.append({'representation':representation,'seed':seed,'backend':backend,'order':order_name,'prefix':cap,
                            'thresholds':chosen[backend],**metrics,'representation_diagnostics':diags[cap],'runtime':runtime})
                        for event in sealed.events:
                            ledgers.append({'representation':representation,'seed':seed,'backend':backend,'order':order_name,'prefix':cap,
                                'sequence':event.sequence,'video_id':event.physical_key.video_id,'physical_track_id':event.physical_key.local_track_id,
                                'kind':event.kind,'known_category_id':event.known_category_id,'token':event.token})
            atomic_json(out/'progress.json',{'representation':representation,'seed':seed,'actual_cases':len(cases),'heldout_opened':True})
            print(json.dumps({'representation':representation,'seed':seed,'cases':len(cases)}),flush=True);del bank,model
            mem={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
            if mem['MemAvailable']<mem['MemTotal']*cfg['system_ram_reserve_fraction'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>cfg['host_planned_peak_bytes'] or time.monotonic()-started>cfg['wall_seconds']:raise RuntimeError('Heldout resource guard')
    aggregates=[]
    metrics={'old_acc':'standard','new_acc':'standard','h_score':'standard','all_acc':'standard','correct_commit_ct':'persistent',
        'false_merge_rate':'persistent','false_split_new_rate':'persistent','wrong_known_assignment_rate':'persistent',
        'wait_unresolved_rate':'persistent','effective_commit_coverage':'persistent','all_novel_wrong_known_rate':'errors'}
    for representation in cfg['representation_variants']:
        names=cfg['baselines_raw'] if representation=='A0_RAW' else [cfg['learned_representation_backend']]
        for backend in names:
            for cap in cfg['prefixes']:
                rows=[r for r in cases if (r['representation'],r['backend'],r['prefix'])==(representation,backend,cap)]
                seeds=sorted({r['seed'] for r in rows if r['seed'] is not None})
                row={'scope':'TAO_TRAIN_GT_CONTROLLED_HELDOUT','representation':representation,'backend':backend,'prefix':cap,'seeds':len(seeds) or 1,'orders':4}
                for metric,section in metrics.items():
                    # Std across training seed means, not treating orders as independent seeds.
                    means=[float(np.mean([r[section][metric] for r in rows if r['seed']==s])) for s in seeds] if seeds else [float(np.mean([r[section][metric] for r in rows]))]
                    row[metric+'_mean']=float(np.mean(means));row[metric+'_seed_std_ddof0']=float(np.std(means))
                for metric in ('cross_video_rank1_macro','known_vs_pseudo_novel_auroc','effective_spectral_rank','cross_video_recall_at_5_macro','cross_video_recall_at_10_macro'):
                    vals=[r['representation_diagnostics'][metric] for r in rows];row[metric+'_mean']=float(np.mean(vals))
                aggregates.append(row)
    path=out/'sealed_predictions.parquet';pq.write_table(pa.Table.from_pylist(ledgers),path)
    receipt={'schema_version':'trackocd.core.main-representation-heldout.v1','status':'COMPLETE_REAL_TRAIN_HELDOUT_NOT_PREDICTED_TRACK_RESULT',
        'preregistration_commit':args.preregistration_commit,'config_sha256':sha256_file(cp),'cases':cases,'aggregate':aggregates,'modules':modules,
        'calibrations':calibrations,'orders':orders,'known_gt':207,'pseudo_novel_gt':16,'PHE':cfg['PHE'],
        'all_seeds_orders_prefixes_kept':True,'all_methods_same_visual_cache':True,'source_sha256':{n:sha256_file(ROOT/n) for n in sources},
        'feature_manifest_sha256':sha256_file(ROOT/'outputs/trackocd_core/features/train_first_v1/manifest.json'),
        'private_ledger':{'path':str(path.relative_to(ROOT)),'rows':len(ledgers),'bytes':path.stat().st_size,'sha256':sha256_file(path)},
        'GT_or_Hungarian_in_prediction_memory':False,'heldout_used_to_redesign_method':False,'val_or_test_access':False,'histories_preserved':True,
        'resources':{'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_workers':1,'gpu_used':False}}
    atomic_json(out/'results.json',receipt);atomic_json(ROOT/'outputs/trackocd_core/TRAIN_FIRST_REPRESENTATION_HELDOUT_RESULT.json',receipt)
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(aggregates[0]),lineterminator='\n');writer.writeheader();writer.writerows(aggregates)
    atomic_write_text(ROOT/'outputs/trackocd_core/REPRESENTATION_ABLATION.csv',buffer.getvalue())
    baseline=[r for r in aggregates if r['representation']=='A0_RAW'];buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(baseline[0]),lineterminator='\n');writer.writeheader();writer.writerows(baseline)
    atomic_write_text(ROOT/'outputs/trackocd_core/TRAIN_BASELINE_COMPARISON.csv',buffer.getvalue())
    print(json.dumps({'status':receipt['status'],'cases':len(cases),'resources':receipt['resources'],'heldout_p16':[r for r in aggregates if r['prefix']==16]}),flush=True)


if __name__=='__main__':main()
