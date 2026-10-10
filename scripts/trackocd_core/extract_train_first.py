#!/usr/bin/env python3
"""Frozen DINOv2 image-grouped encoding, owned workers, compact FP16.

Only current first chronological prefixes; no crop images, per-target JSON,
detector, tracker, optimizer, Val/Test or GT-driven predicted filtering.
Partial output is retained after any failure, never overwritten or retried.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import datetime as dt
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file
from scripts.trackocd_core.audit_assets import DATASET


def inputs():
    cp=ROOT/'configs/trackocd_core/training_split.json';cfg=json.loads(cp.read_text())
    plan_path=ROOT/cfg['output_directory']/'selection_plan.json';plan=json.loads(plan_path.read_text())
    audit=json.loads((ROOT/'outputs/trackocd_core/TRAIN_ONLY_CATEGORY_SPLIT_AUDIT.json').read_text())
    if sha256_file(plan_path)!=audit['private_plan']['sha256']: raise ValueError('Registered Train plan changed')
    for name,sha in plan['sources'].items():
        if sha256_file(ROOT/name)!=sha: raise ValueError('T0 source/config identity changed: '+name)
    visual=json.loads((ROOT/cfg['visual_protocol']).read_text())
    return cfg,plan,visual


def guard(cfg,started):
    info={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if info['MemAvailable']<info['MemTotal']*cfg['system_ram_reserve_fraction']: raise RuntimeError('RAM headroom guard')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>cfg['host_planned_peak_bytes']: raise RuntimeError('Owned worker RSS guard')
    if time.monotonic()-started>cfg['feature_wall_seconds']: raise RuntimeError('Registered wall-time guard')
    if os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize<cfg['disk_reserve_bytes']: raise RuntimeError('Disk reserve guard')


def choose_gpus(cfg):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
    candidates=[]
    for line in text.splitlines():
        uuid,free,util=[s.strip() for s in line.split(',')]
        if int(free)*2**20 >=cfg['gpu_planned_peak_bytes']+cfg['gpu_free_reserve_bytes'] and int(util)<10:
            candidates.append((int(free),uuid))
    candidates.sort(reverse=True)
    if len(candidates)<cfg['worker_count']: raise RuntimeError('Wait for safe free GPU capacity; no foreign intervention')
    return [uuid for free,uuid in candidates[:cfg['worker_count']]]


def worker(index):
    started=time.monotonic();cfg,plan,visual=inputs();cache=ROOT/cfg['cache_directory'];guard(cfg,started)
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from scripts.trackocd_v2.build_common_features import crop_box
    repo=ROOT/visual['local_repository']; checkpoint=ROOT/visual['checkpoint']
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=visual['upstream_commit']:
        raise ValueError('Pinned DINO source differs')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip() or sha256_file(checkpoint)!=visual['checkpoint_sha256']:
        raise ValueError('Frozen source/weight mismatch')
    torch.set_num_threads(1);torch.manual_seed(1027)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(repo));from dinov2.hub.backbones import dinov2_vitb14
    model=dinov2_vitb14(pretrained=False)
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    model.load_state_dict(state,strict=True);del state
    model.eval().requires_grad_(False).to('cuda:0')
    transform=transforms.Compose([transforms.Resize(tuple(visual['resize']),interpolation=Image.Resampling.BILINEAR),
        transforms.ToTensor(),transforms.Normalize(visual['normalization_mean'],visual['normalization_std'])])
    assignment=json.loads((cache/'assignment.json').read_text());rows=assignment['rows']
    videos=sorted({r['video_id'] for r in rows});owned={v for i,v in enumerate(videos) if i%cfg['worker_count']==index}
    visual_out=np.load(cache/'observations.npy',mmap_mode='r+');geometry=np.load(cache/'geometry.npy',mmap_mode='r+')
    by_image=defaultdict(list)
    for r in rows:
        if r['video_id'] in owned:
            for j,obs in enumerate(r['observations']): by_image[obs['image_id']].append((r['observation_offset']+j,obs))
    completed=0;forwards=0;image_hashes=[];reused=0
    for image_id,items in list(by_image.items()):
        need=[]
        for offset,obs in items:
            if np.isfinite(visual_out[offset]).all():completed+=1;reused+=1
            else:need.append((offset,obs))
        if need:by_image[image_id]=need
        else:del by_image[image_id]
    for image_id,items in sorted(by_image.items()):
        guard(cfg,started); path=DATASET/'frames'/items[0][1]['image_path']
        if not path.resolve().is_relative_to((DATASET/'frames/train').resolve()):raise ValueError('Only registered Train pixels')
        # Hash the exact file bytes used by the encoder; check it has not changed.
        before=path.stat();sha=sha256_file(path);tensors=[];locations=[]
        with Image.open(path) as raw:
            rgb=raw.convert('RGB');w,h=rgb.size
            unique={}
            for offset,obs in items:
                key=tuple(obs['bbox_xyxy']);unique.setdefault(key,[]).append(offset)
            for box,offsets in unique.items():
                tensors.append(transform(crop_box(rgb,box,visual['crop_context_fraction_per_side'])))
                locations.append(offsets)
                for offset in offsets: geometry[offset]=np.clip(np.asarray(box)/[w,h,w,h],0,1)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Train image changed during encoding')
        for first in range(0,len(tensors),cfg['feature_batch_size']):
            guard(cfg,started)
            if torch.cuda.mem_get_info()[0]<cfg['gpu_free_reserve_bytes']:raise RuntimeError('GPU free reserve guard')
            with torch.inference_mode():
                value=model.forward_features(torch.stack(tensors[first:first+cfg['feature_batch_size']]).to('cuda:0'))['x_norm_clstoken']
                value=torch.nn.functional.normalize(value,dim=-1).cpu().numpy().astype(np.float16)
            for descriptor,offsets in zip(value,locations[first:first+cfg['feature_batch_size']]):
                for offset in offsets: visual_out[offset]=descriptor;completed+=1
            forwards+=1
            if torch.cuda.max_memory_reserved()>cfg['gpu_planned_peak_bytes']:raise RuntimeError('Registered GPU peak guard')
        image_hashes.append({'image_id':image_id,'bytes':before.st_size,'sha256':sha})
        if len(image_hashes)%50==0:
            visual_out.flush();geometry.flush()
            atomic_json(cache/f'worker{index}_progress.json',{'completed_observations':completed,'encoded_images':len(image_hashes),'seconds':time.monotonic()-started})
    visual_out.flush();geometry.flush()
    expected=sum(len(r['observations']) for r in rows if r['video_id'] in owned)
    if completed!=expected or any(p.requires_grad for p in model.parameters()):raise ValueError('Worker universe/frozen-state guard')
    atomic_json(cache/f'worker{index}_images.json',image_hashes)
    atomic_json(cache/f'worker{index}_done.json',{'worker':index,'completed_observations':completed,'encoded_images':len(image_hashes),'batch_forwards':forwards,
        'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'wall_seconds':time.monotonic()-started,'reused_observations':reused,
        'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'frozen_encoder':True,'optimizer_used':False})


def supervise(preregistration_commit):
    started=time.monotonic();cfg,plan,visual=inputs();cache=ROOT/cfg['cache_directory'];guard(cfg,started)
    if cache.exists():raise ValueError('Preserve any completed or partial main cache; no automatic overwrite/retry')
    branch='refs/heads/codex/trackocd-core-training-limited-masa'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin',branch],text=True,timeout=25).split()[0]
    if remote!=preregistration_commit:raise ValueError('Exact remote T0 registration required before extraction')
    for name in ('configs/trackocd_core/training_split.json','scripts/trackocd_core/extract_train_first.py','src/trackocd_core/train_first_cache.py'):
        registered=subprocess.check_output(['git','show',f'{preregistration_commit}:{name}'])
        if registered!=(ROOT/name).read_bytes():raise ValueError('Registration source changed')
    gpus=choose_gpus(cfg)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows=[];offset=0
    for row in plan['rows']:
        rows.append({**row,'observation_offset':offset});offset+=len(row['observations'])
    cache.mkdir(parents=True)
    np.lib.format.open_memmap(cache/'observations.npy',mode='w+',dtype=np.float16,shape=(offset,768))[:] = np.nan
    np.lib.format.open_memmap(cache/'geometry.npy',mode='w+',dtype=np.float32,shape=(offset,4))[:] = np.nan
    # Reuse exact compatible current-prefix observations, regardless of their
    # old simulation role; descriptors contain no semantic labels. Preserve
    # original payloads and disclose old missing per-image raw-byte digests.
    from src.trackocd_core.features import CompactGTFeasibilityCache
    lookup={};reuse_sources=[]
    for name in ('gt_train_known_smoke','gt_train_known_pilot'):
        oldroot=ROOT/'outputs/trackocd_core/features'/name
        old=CompactGTFeasibilityCache(oldroot)
        if old.manifest['config']['sha256']!=sha256_file(ROOT/cfg['visual_protocol']) or old.manifest['annotation']['sha256']!=plan['annotation_sha256']:
            continue
        oldrows=pq.read_table(oldroot/'index.parquet').to_pylist()
        oldvisual=np.load(oldroot/'observations.npy',mmap_mode='r');oldgeometry=np.load(oldroot/'geometry.npy',mmap_mode='r')
        for r in oldrows:
            for j in range(r['observation_count']):
                key=(r['image_paths'][j],r['image_ids'][j],r['frame_ids'][j],tuple(r['boxes_xyxy'][j]))
                lookup[key]=(np.array(oldvisual[r['observation_offset']+j]),np.array(oldgeometry[r['observation_offset']+j]))
        reuse_sources.append({'cache':name,'manifest_sha256':sha256_file(oldroot/'manifest.json'),'payloads':old.manifest['payloads']})
    newvisual=np.load(cache/'observations.npy',mmap_mode='r+');newgeometry=np.load(cache/'geometry.npy',mmap_mode='r+');reused=0
    for r in rows:
        for j,o in enumerate(r['observations']):
            key=(o['image_path'],o['image_id'],o['frame_id'],tuple(o['bbox_xyxy']))
            if key in lookup:
                newvisual[r['observation_offset']+j],newgeometry[r['observation_offset']+j]=lookup[key];reused+=1
    newvisual.flush();newgeometry.flush();del newvisual,newgeometry,lookup
    atomic_json(cache/'assignment.json',{'rows':rows,'preregistration_commit':preregistration_commit})
    children=[];logs=[];error=None
    try:
        for i,uuid in enumerate(gpus):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
            log=(cache/f'worker{i}.log').open('x');logs.append(log)
            children.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT))
        print(json.dumps({'stage':'FROZEN_FEATURE_ENCODING_NOT_ADAPTER_TRAINING','gpus':gpus,'workers':len(children),'observations':offset}),flush=True)
        while any(p.poll() is None for p in children):
            guard(cfg,started)
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Owned feature worker failed; preserve partial evidence')
            time.sleep(2)
        if any(p.returncode!=0 for p in children):raise RuntimeError('Owned worker terminal failure')
        summaries=[json.loads((cache/f'worker{i}_done.json').read_text()) for i in range(len(children))]
        if sum(r['completed_observations'] for r in summaries)!=offset:raise ValueError('Missing observation universe')
        if not np.isfinite(np.load(cache/'observations.npy',mmap_mode='r')).all():raise ValueError('Incomplete descriptor payload')
        index=[];labels=[]
        for row in rows:
            index.append({k:row[k] for k in ('key','video_id','physical_track_id','observation_offset')}|{
                'observation_count':len(row['observations']),'frame_ids':[o['frame_id'] for o in row['observations']],
                'image_ids':[o['image_id'] for o in row['observations']],'boxes_xyxy':[o['bbox_xyxy'] for o in row['observations']],
                'quality':[1.]*len(row['observations'])})
            labels.append({k:row[k] for k in ('key','category_id','partition','simulation_role')})
        pq.write_table(pa.Table.from_pylist(index),cache/'index.parquet');pq.write_table(pa.Table.from_pylist(labels),cache/'train_labels.parquet')
        payloads={name:{'bytes':(cache/name).stat().st_size,'sha256':sha256_file(cache/name)} for name in
                  ('observations.npy','geometry.npy','index.parquet','train_labels.parquet')}
        if sum(r['bytes'] for r in payloads.values())>cfg['new_feature_payload_ceiling_bytes']:raise ValueError('Payload cap')
        manifest={'schema_version':'trackocd.core.train-first-features.v1','status':'COMPLETE_TRAIN_KNOWN_MAIN_FEATURES_NOT_TRAINING_OR_PREDICTED_RESULT',
            'source_role':'TAO Train Known GT main training, not predicted/end-to-end','tracks':len(rows),'observations':offset,
            'payloads':payloads,'scope_sha256':sha256_file(ROOT/cfg['scope_amendment']),'config_sha256':sha256_file(ROOT/'configs/trackocd_core/training_split.json'),
            'selection_plan_sha256':sha256_file(ROOT/cfg['output_directory']/'selection_plan.json'),
            'visual_protocol_sha256':sha256_file(ROOT/cfg['visual_protocol']),'DINO_checkpoint_sha256':visual['checkpoint_sha256'],
            'preregistration_commit':preregistration_commit,'workers':summaries,'prefixes':cfg['prefixes'],'unit_GT_quality_not_detector_score':True,
            'reused_exact_protocol_observations':reused,'reuse_source_manifests':reuse_sources,
            'historical_cache_reuse':'Exact protocol/checkpoint/annotation/image_id/path/bbox/frame metadata and payload-hashed smoke/pilot observations reused. Historical per-image raw-byte digests unavailable, disclosed rather than newly claimed.',
            'duplicate_current_image_bbox_encoding_avoided':True,'one_image_decode_per_worker_image':True,
            'optimizer_used':False,'val_or_test_access':False,'historical_outputs_overwritten':False,'external_process_interference':False,
            'resources':{'wall_seconds':time.monotonic()-started,'worker_count':len(children)}}
        atomic_json(cache/'manifest.json',manifest);atomic_json(cache/'.done',{'manifest_sha256':sha256_file(cache/'manifest.json')})
        from src.trackocd_core.train_first_cache import TrainFirstCache
        TrainFirstCache(cache)
        atomic_json(ROOT/'outputs/trackocd_core/TRAIN_FIRST_FEATURE_RESULT.json',manifest)
        print(json.dumps({'status':manifest['status'],'tracks':len(rows),'observations':offset,'resources':manifest['resources']}),flush=True)
    except BaseException as exc:
        error=str(exc)
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
        raise
    finally:
        for log in logs:log.close()
        atomic_json(cache/'supervisor.json',{'preregistration_commit':preregistration_commit,'error':error,
            'owned_child_returncodes':[p.returncode for p in children],'owned_child_pids':[p.pid for p in children],
            'wall_seconds':time.monotonic()-started,'no_foreign_process_interference':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--preregistration-commit');a=p.parse_args()
    if a.worker is not None:worker(a.worker)
    else:
        if not a.preregistration_commit:p.error('--preregistration-commit required')
        supervise(a.preregistration_commit)
