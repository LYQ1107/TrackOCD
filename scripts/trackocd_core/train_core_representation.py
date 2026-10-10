#!/usr/bin/env python3
"""Real independent main fits, exact remote preregistration before optimizer."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file
from scripts.trackocd_core.extract_train_first import choose_gpus, inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True)
    p.add_argument('--phase',choices=('representation','evidence'),default='representation');args=p.parse_args()
    base_cp=ROOT/'configs/trackocd_core/representation_training.json';base=json.loads(base_cp.read_text())
    cp=base_cp;cfg=base;parent_result=None
    if args.phase=='evidence':
        cp=ROOT/'configs/trackocd_core/evidence_training.json';ev=json.loads(cp.read_text())
        parent_result=json.loads((ROOT/base['output_directory']/'training_receipt.json').read_text())
        shared_weight=base['geometry_weights'][parent_result['shared_geometry_family']]
        cfg={**base,'models':ev['models'],'geometry_weights':{n:shared_weight for n in ev['models']},
             'output_directory':'outputs/trackocd_core/core_training/train_first_v1/evidence'}
    split=json.loads((ROOT/cfg['split']).read_text());out=ROOT/cfg['output_directory']
    registered_split,plan,visual_protocol=inputs()
    if registered_split!=split:raise ValueError('Registered T0 split differs')
    if out.exists():raise ValueError('Preserve any independent main fits; no overwrite/retry')
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=args.preregistration_commit:raise ValueError('Exact remote training source registration required')
    names=('configs/trackocd_core/representation_training.json','configs/trackocd_core/evidence_training.json',
           'scripts/trackocd_core/train_core_representation.py','src/trackocd_core/train_first_representation.py')
    for name in names:
        if subprocess.check_output(['git','show',f'{args.preregistration_commit}:{name}'])!=(ROOT/name).read_bytes():raise ValueError('Training source changed')
    cache_root=ROOT/split['cache_directory'];supervisor=json.loads((cache_root/'supervisor.json').read_text())
    if supervisor['error'] is not None or any(c!=0 for c in supervisor['owned_child_returncodes']):raise ValueError('Only successfully sealed full feature cache')
    gpu_cfg={**split,'worker_count':1,'gpu_planned_peak_bytes':cfg['gpu_planned_peak_bytes']}
    uuid=choose_gpus(gpu_cfg)[0];os.environ['CUDA_VISIBLE_DEVICES']=uuid;os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    from torch.nn import functional as F
    from src.trackocd_core.train_first_cache import TrainFirstCache
    from src.trackocd_core.train_first_representation import make_model,BalancedBatchSampler,forward_views,development_metrics
    from src.trackocd_core.representation import cross_video_category_loss
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    started=time.monotonic();cache=TrainFirstCache(cache_root)
    if (cache.manifest['config_sha256']!=sha256_file(ROOT/cfg['split']) or
        cache.manifest['selection_plan_sha256']!=sha256_file(ROOT/split['output_directory']/'selection_plan.json') or
        cache.manifest['visual_protocol_sha256']!=sha256_file(ROOT/split['visual_protocol']) or
        cache.manifest['DINO_checkpoint_sha256']!=visual_protocol['checkpoint_sha256']):
        raise ValueError('Frozen feature/encoder/protocol lineage differs')
    labels=pq.read_table(cache_root/'train_labels.parquet').to_pylist()
    registered={r['key']:r for r in plan['rows']}
    legal_known=set(json.loads((ROOT/split['roles']).read_text())['known_ids'])
    if set(r['key'] for r in labels)!=set(registered) or any(r['category_id'] not in legal_known for r in labels):
        raise ValueError('Entire Train Known-only supervision universe required')
    for r in labels:
        if any(r[k]!=registered[r['key']][k] for k in ('category_id','partition','simulation_role')):
            raise ValueError('Registered category/video partition changed')
    routes={r['key']:r for r in cache.rows};labels=[{**r,'video_id':routes[r['key']]['video_id'],'physical_track_id':routes[r['key']]['physical_track_id']} for r in labels]
    fit=[r for r in labels if r['partition']=='representation_fit'];known_ids=sorted({r['category_id'] for r in fit})
    if len(fit)!=1305 or len(known_ids)!=15:raise ValueError('Entire independent registered fitting universe required')
    out.mkdir(parents=True);fits=[];checkpoint_bytes=0
    raw_development=development_metrics(cache,labels,known_ids)
    atomic_json(out/'raw_development.json',raw_development)

    def guard():
        mem={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if mem['MemAvailable']<mem['MemTotal']*cfg['system_ram_reserve_fraction']:raise RuntimeError('RAM reserve guard')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>cfg['host_planned_peak_bytes']:raise RuntimeError('Host RSS guard')
        if time.monotonic()-started>cfg['training_wall_seconds']:raise RuntimeError('Training wall guard')
        if torch.cuda.max_memory_reserved()>cfg['gpu_planned_peak_bytes'] or torch.cuda.mem_get_info()[0]<cfg['gpu_free_reserve_bytes']:raise RuntimeError('GPU reserve/peak guard')
        if os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize<split['disk_reserve_bytes']:raise RuntimeError('Disk reserve guard')
    for seed in cfg['seeds']:
        for name in cfg['models']:
            guard();torch.manual_seed(seed);model=make_model(name).to('cuda:0').train()
            # Initialization identity of shared adapter, independent of sampler RNG.
            import hashlib
            initial=hashlib.sha256()
            for k,v in sorted(model.adapter.state_dict().items()):initial.update(k.encode());initial.update(v.detach().cpu().numpy().tobytes())
            sampler=BalancedBatchSampler(fit,seed,cfg['batch_categories'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            trace=[];checkpoints=[];seen=set();run_started=time.monotonic();mask_positive=mask_negative=0;batch_digest=hashlib.sha256()
            for step in range(1,cfg['steps']+1):
                if step%25==1:guard()
                batch=sampler.draw();cap=cfg['prefix_cycle'][(step-1)%5];views=[cache.get_prefix(r['key'],cap) for r in batch]
                batch_digest.update(json.dumps([(r['key'],len(v.visual),cap) for r,v in zip(batch,views)]).encode())
                seen.update(r['key'] for r in batch)
                category=torch.tensor([r['category_id'] for r in batch],device='cuda:0')
                videos=torch.tensor([r['video_id'] for r in batch],device='cuda:0')
                physical=torch.tensor([r['physical_track_id'] for r in batch],device='cuda:0')
                embedding=forward_views(model,views,'cuda:0')
                category_loss=cross_video_category_loss(embedding,category,videos,physical,cfg['contrastive_temperature'])
                teacher=torch.tensor(np.stack([v.weighted_mean() for v in views]),device='cuda:0')
                geometry=F.mse_loss(embedding@embedding.T,teacher@teacher.T)
                loss=category_loss+cfg['geometry_weights'][name]*geometry
                if not torch.isfinite(loss):raise ValueError('Nonfinite main fit loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm']);optimizer.step()
                positives=int(((category[:,None]==category[None,:])&(videos[:,None]!=videos[None,:])).sum().item())
                negatives=int((category[:,None]!=category[None,:]).sum().item());mask_positive+=positives;mask_negative+=negatives
                trace.append({'step':step,'prefix_cap':cap,'actual_observations':sum(len(v.visual) for v in views),'category_loss':float(category_loss.detach()),
                              'geometry_loss':float(geometry.detach()),'total_loss':float(loss.detach()),'positive_pairs':positives,'negative_pairs':negatives})
                if step in cfg['checkpoint_steps']:
                    model.eval();dev=development_metrics(cache,labels,known_ids,model,'cuda:0');model.train()
                    checkpoint=out/f'{name}_seed{seed}_step{step}.pt';tmp=checkpoint.with_suffix('.pt.tmp')
                    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'model':name,'seed':seed,'step':step,
                                'config_sha256':sha256_file(cp),'feature_manifest_sha256':sha256_file(cache_root/'manifest.json')},tmp);tmp.rename(checkpoint)
                    checkpoint_bytes+=checkpoint.stat().st_size
                    if checkpoint_bytes>cfg['new_checkpoint_payload_ceiling_bytes']:raise ValueError('Checkpoint payload guard')
                    checkpoints.append({'step':step,'checkpoint':{'path':str(checkpoint.relative_to(ROOT)),'bytes':checkpoint.stat().st_size,'sha256':sha256_file(checkpoint)},'development':dev})
                    atomic_json(out/'progress.json',{'stage':'REAL_SEMANTIC_ADAPTER_TRAINING','seed':seed,'model':name,'completed_steps':step,'maximum_steps':cfg['steps'],'completed_fits':len(fits)})
                    print(json.dumps({'seed':seed,'model':name,'step':step,'category_loss':trace[-1]['category_loss'],'development_score':dev['registered_checkpoint_score']}),flush=True)
            usable=[c for c in checkpoints if c['development']['registered_checkpoint_score'] is not None]
            if not usable:raise ValueError('No supported development checkpoint selection; preserve results')
            if parent_result is None:
                selected=max(usable,key=lambda c:(c['development']['registered_checkpoint_score'],-c['step']))
            else:
                reference_step=parent_result['selected_geometry'][str(seed)]['step']
                selected=next(c for c in usable if c['step']==reference_step)
                reference=next(f for f in parent_result['fits'] if f['seed']==seed and f['model']==parent_result['shared_geometry_family'])
                if batch_digest.hexdigest()!=reference['batch_identity_prefix_sha256']:raise ValueError('A1/A2 actual observations or pairs differ')
                if initial.hexdigest()!=reference['adapter_initial_state_sha256']:raise ValueError('A1/A2 adapter init differs')
            fits.append({'model':name,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'steps':cfg['steps'],
                         'fit_tracks_available':len(fit),'fit_tracks_actually_seen':len(seen),'fit_classes':known_ids,
                         'adapter_initial_state_sha256':initial.hexdigest(),'batch_identity_prefix_sha256':batch_digest.hexdigest(),
                         'positive_pairs':mask_positive,'negative_pairs':mask_negative,
                         'trace':trace,'checkpoints':checkpoints,'selected_checkpoint':selected,'wall_seconds':time.monotonic()-run_started})
            atomic_json(out/'fits_partial.json',fits);del optimizer,model
    selected_geometry={}
    family_scores={name:float(np.mean([f['selected_checkpoint']['development']['registered_checkpoint_score']
                                      for f in fits if f['model']==name])) for name in cfg['models']}
    family=max(cfg['models'],key=lambda name:(family_scores[name],-cfg['geometry_weights'][name]))
    for seed in cfg['seeds']:
        options=[f for f in fits if f['seed']==seed]
        best=next(f for f in options if f['model']==family)
        selected_geometry[str(seed)]={'model':best['model'],'geometry_weight':cfg['geometry_weights'][best['model']],
                                     'step':best['selected_checkpoint']['step'],
                                     'checkpoint':best['selected_checkpoint']['checkpoint']}
        assert len({f['adapter_initial_state_sha256'] for f in options})==1
    receipt={'schema_version':'trackocd.core.main-representation-result.v1','status':'COMPLETE_REAL_TRAIN_ONLY_MAIN_FITS_NOT_SCIENTIFIC_PASS',
             'preregistration_commit':args.preregistration_commit,'config_sha256':sha256_file(cp),'feature_manifest_sha256':sha256_file(cache_root/'manifest.json'),
             'fit_tracks':len(fit),'fit_classes':known_ids,'fits':fits,'selected_geometry':selected_geometry,
             'shared_geometry_family':family,'development_geometry_family_mean_scores':family_scores,'raw_development':raw_development,
             'checkpoint_bytes':checkpoint_bytes,'source_sha256':{n:sha256_file(ROOT/n) for n in names},
             'phase':args.phase,'matched_A1_selected_checkpoint_steps':parent_result is not None,
             'new_main_protocol_not_R2':True,'historical_outputs_overwritten':False,'DINO_detector_tracker_trained':False,
             'policy_or_heldout_features_used_for_fit':False,'final_heldout_evaluated':False,'val_or_test_access':False,
             'resources':{'wall_seconds':time.monotonic()-started,'gpu_uuid':uuid,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                          'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
    public_name='TRAIN_FIRST_REPRESENTATION_RESULT.json' if args.phase=='representation' else 'TRAIN_FIRST_EVIDENCE_RESULT.json'
    atomic_json(out/'training_receipt.json',receipt);atomic_json(ROOT/'outputs/trackocd_core'/public_name,receipt)
    print(json.dumps({'status':receipt['status'],'fits':len(fits),'resources':receipt['resources']}),flush=True)

if __name__=='__main__':main()
