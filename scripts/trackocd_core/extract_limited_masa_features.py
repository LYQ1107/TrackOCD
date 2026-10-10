#!/usr/bin/env python3
"""One-image/many-crop frozen DINO, all predicted IDs, owned atomic workers.

No annotations/roles/Novel words/model supervision, detector or tracker input.
Full continuation requires a sealed successful semantic integration result.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file
from src.trackocd_core.physical_qualification import completed_video,val_image
from src.trackocd_core.limited_masa_features import select_all_prefixes,completed_feature

CONFIG=ROOT/'configs/trackocd_core/limited_masa_features.json'
SOURCES=('configs/trackocd_core/limited_masa_features.json','scripts/trackocd_core/extract_limited_masa_features.py','src/trackocd_core/limited_masa_features.py')


def inputs():
    cfg=json.loads(CONFIG.read_text());common_path=ROOT/cfg['common_visual_config']
    for path,sha in ((common_path,cfg['common_visual_config_sha256']),
                     (ROOT/cfg['prediction_run']/'prediction_manifest.json',cfg['prediction_manifest_sha256']),
                     (ROOT/cfg['private_metadata_plan'],cfg['private_metadata_plan_sha256'])):
        if sha256_file(path)!=sha:raise ValueError('Frozen input identity differs')
    freeze=json.loads((ROOT/cfg['model_freeze']).read_text())
    if freeze['status']!='FROZEN_TRAIN_ONLY_MODELS_AND_OPERATING_POINTS':raise ValueError('Models must actually freeze before predicted encoding')
    for path,sha in freeze['protected_sha256'].items():
        if sha256_file(ROOT/path)!=sha:raise ValueError('Frozen Train model/protocol identity changed')
    return cfg,json.loads(common_path.read_text()),json.loads((ROOT/cfg['private_metadata_plan']).read_text())


def barrier():
    def audit(event,args):
        if event in {'socket.connect','socket.getaddrinfo','socket.sendto','subprocess.Popen','os.system'}:raise PermissionError('Frozen predicted encoder prohibits network/child work')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=os.fsdecode(args[0])
            if ('/TAO-Amodal/annotations/' in p or '/frames/test/' in p or p.endswith(('/roles.json','/train_labels.parquet','/evaluator_only_geometry_join.json'))):raise PermissionError('Predicted encoder has no GT/role/Test access')
    sys.addaudithook(audit)


def worker(index,assignment_path):
    import numpy as np
    cfg,common,plan=inputs();assignment=json.loads(Path(assignment_path).read_text());out=ROOT/cfg['output_directory'];shards=out/'shards'
    if assignment['source_sha256']!={n:sha256_file(ROOT/n) for n in SOURCES}:raise ValueError('Worker source registration changed')
    repo=ROOT/common['local_repository']
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=common['upstream_commit'] or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():raise ValueError('Pinned frozen DINO source changed')
    if sha256_file(ROOT/common['checkpoint'])!=common['checkpoint_sha256']:raise ValueError('DINO checkpoint differs')
    old_smoke_source=subprocess.check_output(['git','show','f25ba9a323878ed7376cf3c8dec93540071bfffe:scripts/trackocd_core/smoke_predicted_features.py'])
    if old_smoke_source!=(ROOT/'scripts/trackocd_core/smoke_predicted_features.py').read_bytes():raise ValueError('Old reusable smoke source differs')
    owned=set(assignment['worker_videos'][str(index)]);videos=[v for v in plan['videos'] if v['video_id'] in owned]
    barrier();import torch
    from PIL import Image
    from torchvision import transforms
    from scripts.trackocd_v2.build_common_features import crop_box
    sys.path.insert(0,str(repo));from dinov2.hub.backbones import dinov2_vitb14
    started=time.monotonic();torch.set_num_threads(1);torch.manual_seed(1027)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=dinov2_vitb14(pretrained=False);state=torch.load(ROOT/common['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
    model.load_state_dict(state,strict=True);del state;model.eval().requires_grad_(False).to('cuda:0')
    transform=transforms.Compose([transforms.Resize(tuple(common['resize']),interpolation=Image.Resampling.BILINEAR),
        transforms.ToTensor(),transforms.Normalize(common['normalization_mean'],common['normalization_std'])])
    summaries=[];reuse_total=0
    def guard():
        mem={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
        if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>cfg['host_planned_peak_per_worker_bytes'] or
            mem['MemAvailable']<mem['MemTotal']*cfg['system_RAM_reserve_fraction'] or torch.cuda.max_memory_reserved()>cfg['GPU_planned_peak_bytes'] or
            torch.cuda.mem_get_info()[0]<cfg['GPU_free_reserve_bytes'] or time.monotonic()-started>cfg['wall_seconds']):raise RuntimeError('Owned feature resource guard')
        if os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize<cfg['minimum_disk_free_bytes']:raise RuntimeError('Feature disk reserve guard')
    for video in videos:
        guard();record=completed_video(ROOT/cfg['prediction_run'],video,cfg['prediction_config_sha256'])
        if record is None:raise ValueError('Only completed frozen physical inputs')
        existing=completed_feature(shards,video['video_id'],sha256_file(CONFIG),record['npz_sha256'])
        if existing is not None:summaries.append(existing);continue
        with np.load(ROOT/cfg['prediction_run']/'shards'/record['npz_filename'],allow_pickle=False) as a:
            rows,count=select_all_prefixes(video,a,cfg['maximum_observations'])
            visual=np.full((count,768),np.nan,dtype=np.float16);geometry=np.empty((count,4),np.float32)
            quality=np.empty(count,np.float32);frames=np.empty(count,np.int64);image_ids=np.empty(count,np.int64)
            grouped=defaultdict(list)
            for r in rows:
                for j,o in enumerate(r['observations']):grouped[o['image_position']].append((r['observation_offset']+j,o['source_offset']))
            # Reconstruct the exact tiny source selection,not a guessed mapping.
            smoke_path=ROOT/'outputs/trackocd_core/features/masa_predicted_interface_smoke/receipt.json'
            reused=0
            if video['video_id']==4:
                smoke=json.loads(smoke_path.read_text());payload=smoke_path.parent/'engineering_only.npz'
                old_config=json.loads((ROOT/'configs/trackocd_core/masa_predicted_feature_smoke.json').read_text())
                if (smoke['status']=='PASS_TINY_REAL_PREDICTED_INTERFACE_NOT_PRIMARY_QUALIFICATION' and
                    old_config['common_config_sha256']==cfg['common_visual_config_sha256'] and
                    smoke['input_shard_sha256']==record['npz_sha256'] and smoke['checkpoint_sha256']==common['checkpoint_sha256'] and
                    sha256_file(payload)==smoke['private_payload']['sha256']):
                    from scripts.trackocd_core.smoke_predicted_features import select_observed_prefixes
                    tiny,_=select_observed_prefixes(video,a);lookup={};start=0
                    with np.load(payload,allow_pickle=False) as old:
                        for r in tiny:
                            for j,o in enumerate(r['observations']):lookup[(r['track_id'],o['image']['image_id'],tuple(o['box']))]=old['visual'][start+j].copy()
                            start+=len(r['observations'])
                    for r in rows:
                        for j,o in enumerate(r['observations']):
                            k=(r['physical_track_id'],video['images'][o['image_position']]['image_id'],tuple(a['boxes'][o['source_offset']]))
                            if k in lookup:visual[r['observation_offset']+j]=lookup[k];reused+=1
            encoded_images=0;input_images=[];forwards=0;deduplicated=0
            for position,locations in sorted(grouped.items()):
                guard();image=video['images'][position];path=val_image(Path('/data3/liuyeqiang/TAO-Amodal/frames'),image['image_path'])
                if path.stat().st_size!=image['image_bytes'] or sha256_file(path)!=image['image_sha256']:raise ValueError('Current Val image bytes differ from frozen metadata')
                before=path.stat();unique=defaultdict(list)
                with Image.open(path) as raw:
                    rgb=raw.convert('RGB');w,h=rgb.size
                    for offset,source in locations:
                        geometry[offset]=np.clip(a['boxes'][source]/[w,h,w,h],0,1);quality[offset]=a['score'][source]
                        frames[offset]=image['frame_index'];image_ids[offset]=image['image_id']
                        if not np.isfinite(visual[offset]).all():unique[tuple(a['boxes'][source])].append(offset)
                    boxes=list(unique);deduplicated+=sum(len(v)-1 for v in unique.values())
                    for first in range(0,len(boxes),cfg['batch_size']):
                        chosen=boxes[first:first+cfg['batch_size']];tensors=[transform(crop_box(rgb,box,common['crop_context_fraction_per_side'])) for box in chosen]
                        guard()
                        with torch.inference_mode():
                            value=model.forward_features(torch.stack(tensors).to('cuda:0'))['x_norm_clstoken']
                            value=torch.nn.functional.normalize(value,dim=-1).cpu().numpy().astype(np.float16)
                        for box,z in zip(chosen,value):
                            for offset in unique[box]:visual[offset]=z
                        forwards+=1
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Image changed while encoding')
                encoded_images+=bool(unique);input_images.append({'image_id':image['image_id'],'sha256':image['image_sha256'],'bytes':image['image_bytes']})
            if not np.isfinite(visual).all() or any(p.requires_grad for p in model.parameters()):raise ValueError('Incomplete/unfrozen predicted encoding')
            filename=f"video_{video['video_id']:04d}.{assignment['attempt']}.npz";path=shards/filename;temporary=path.with_suffix('.npz.tmp')
            if path.exists() or temporary.exists():raise ValueError('Never overwrite retained feature attempt')
            offsets=[r['observation_offset'] for r in rows]+[count]
            with temporary.open('xb') as writer:
                np.savez_compressed(writer,visual=visual,geometry=geometry,quality=quality,frame_index=frames,image_id=image_ids,
                    track_id=np.asarray([r['physical_track_id'] for r in rows],dtype=np.int64),offsets=np.asarray(offsets,dtype=np.int64))
                writer.flush();os.fsync(writer.fileno())
            temporary.rename(path)
            result={'status':'SEALED_COMPLETE_ALL_ID_PREFIX_FEATURES','video_id':video['video_id'],'tracks':len(rows),'observations':count,
                'config_sha256':sha256_file(CONFIG),'input_npz_sha256':record['npz_sha256'],'npz_filename':filename,'npz_bytes':path.stat().st_size,'npz_sha256':sha256_file(path),
                'DINO_sha256':common['checkpoint_sha256'],'source_sha256':assignment['source_sha256'],'GT_or_roles_input':False,'unmatched_tracks_removed':False,
                'reused_existing_smoke_observations':reused,'deduplicated_current_image_bbox_crops':deduplicated,'images_encoded':encoded_images,'batch_forwards':forwards,
                'input_images_byte_verified':input_images}
            atomic_json(shards/f"video_{video['video_id']:04d}.complete.json",result);summaries.append(result);reuse_total+=reused
            atomic_json(out/f'worker{index}_progress.json',{'completed_videos':len(summaries),'last_video_id':video['video_id'],'observations':sum(s['observations'] for s in summaries),'seconds':time.monotonic()-started})
            del visual,geometry,quality,frames,image_ids
    atomic_json(out/f"worker{index}_{assignment['attempt']}_done.json",{'videos':len(summaries),'observations':sum(s['observations'] for s in summaries),
        'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'GPU_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'reused_smoke_observations':reuse_total,'optimizer_used':False,'GT_or_roles_input':False})


def supervise(commit,mode):
    cfg,common,plan=inputs();out=ROOT/cfg['output_directory'];started=time.monotonic()
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=commit:raise ValueError('Exact remote feature registration required')
    for n in SOURCES:
        if subprocess.check_output(['git','show',f'{commit}:{n}'])!=(ROOT/n).read_bytes():raise ValueError('Feature source/config registration changed')
    if mode=='full':
        integration=json.loads((out/'integration_result.json').read_text())
        if integration['status']!='PASS_FROZEN_FEATURE_BASELINE_POLICY_EVALUATOR_INTEGRATION':raise ValueError('Actual semantic integration must pass before full encoding')
    videos=plan['videos'][:4] if mode=='integration' else plan['videos'];workers=cfg['workers_integration'] if mode=='integration' else cfg['workers_full']
    mem={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable']-workers*cfg['host_planned_peak_per_worker_bytes']<mem['MemTotal']*cfg['system_RAM_reserve_fraction']:raise RuntimeError('Wait for safe RAM capacity')
    from scripts.trackocd_core.extract_train_first import choose_gpus
    gpus=choose_gpus({'worker_count':workers,'gpu_planned_peak_bytes':cfg['GPU_planned_peak_bytes'],'gpu_free_reserve_bytes':cfg['GPU_free_reserve_bytes']})
    out.mkdir(parents=True,exist_ok=True);(out/'shards').mkdir(exist_ok=True);attempt=uuid.uuid4().hex
    assignment_path=out/f'assignment_{mode}_{attempt}.json'
    assignment={'mode':mode,'attempt':attempt,'preregistration_commit':commit,'source_sha256':{n:sha256_file(ROOT/n) for n in SOURCES},
        'model_freeze_sha256':sha256_file(ROOT/cfg['model_freeze']),
        'worker_videos':{str(i):[v['video_id'] for j,v in enumerate(videos) if j%workers==i] for i in range(workers)}}
    atomic_json(assignment_path,assignment);children=[];logs=[];error=None
    try:
        for i,gpu in enumerate(gpus):
            log=(out/f'worker{i}_{attempt}.log').open('x');logs.append(log)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
            children.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(i),'--assignment',str(assignment_path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        print(json.dumps({'stage':'FROZEN_DINO_PREDICTED_FEATURE_ENCODING_NOT_DETECTOR_TRACKER_OR_TRAINING','mode':mode,'workers':workers,'GPU_UUIDs':gpus}),flush=True)
        while any(p.poll() is None for p in children):
            if time.monotonic()-started>cfg['wall_seconds'] or any(p.poll() not in (None,0) for p in children):raise RuntimeError('Owned feature worker/wall failure; preserve attempts')
            size=sum(p.stat().st_size for p in (out/'shards').iterdir() if p.is_file())
            if size>cfg['new_feature_storage_ceiling_bytes']:raise RuntimeError('Formal feature storage ceiling')
            time.sleep(2)
        if any(p.returncode!=0 for p in children):raise RuntimeError('Feature worker terminal failure')
        records=[]
        manifest=json.loads((ROOT/cfg['prediction_run']/'prediction_manifest.json').read_text());sources={r['video_id']:r for r in manifest['videos']}
        for video in videos:
            r=completed_feature(out/'shards',video['video_id'],sha256_file(CONFIG),sources[video['video_id']]['npz_sha256'])
            if r is None:raise ValueError('Missing atomic feature shard')
            records.append(r)
        summary={'status':'COMPLETE_ALL_ID_PREFIX_FEATURES_INTEGRATION_ONLY' if mode=='integration' else 'COMPLETE_FULL_LIMITED_MASA_ALL_ID_PREFIX_FEATURES',
            'mode':mode,'preregistration_commit':commit,'config_sha256':sha256_file(CONFIG),'model_freeze_sha256':assignment['model_freeze_sha256'],
            'source_sha256':assignment['source_sha256'],'videos':len(records),'tracks':sum(r['tracks'] for r in records),'observations':sum(r['observations'] for r in records),
            'payload_bytes':sum(r['npz_bytes'] for r in records),'reused_smoke_observations':sum(r['reused_existing_smoke_observations'] for r in records),
            'workers':[json.loads((out/f'worker{i}_{attempt}_done.json').read_text()) for i in range(workers)],'records':records,
            'optimizer_used':False,'detector_tracker_inference':False,'GT_or_role_input':False,'unmatched_tracks_removed':False,'TAO_Test_access':False,
            'wall_seconds':time.monotonic()-started,'limited_not_strong_frontend':True,'annotated_cadence_not_dense_frame_online':True}
        if mode=='full' and (summary['videos'],summary['tracks'],summary['observations'])!=(988,304561,1294110):raise ValueError('Full all-ID universe differs')
        atomic_json(out/f'{mode}_manifest.json',summary)
        public={k:v for k,v in summary.items() if k!='records'}
        public['private_manifest_sha256']=sha256_file(out/f'{mode}_manifest.json')
        atomic_json(ROOT/f'outputs/trackocd_core/LIMITED_MASA_FEATURE_{mode.upper()}_RESULT.json',public)
        print(json.dumps({k:public[k] for k in ('status','videos','tracks','observations','payload_bytes','wall_seconds')}),flush=True)
    except Exception as exc:
        error=repr(exc)
        # Only child groups freshly created and retained by this supervisor.
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
        raise
    finally:
        for log in logs:log.close()
        atomic_json(out/f'supervisor_{mode}_{attempt}.json',{'attempt':attempt,'mode':mode,'error':error,'owned_child_returncodes':[p.returncode for p in children],
            'foreign_process_interference':False,'wall_seconds':time.monotonic()-started})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preregistration-commit');parser.add_argument('--mode',choices=('integration','full'))
    parser.add_argument('--worker',type=int);parser.add_argument('--assignment');args=parser.parse_args()
    if args.worker is not None:worker(args.worker,args.assignment)
    elif args.preregistration_commit and args.mode:supervise(args.preregistration_commit,args.mode)
    else:parser.error('Specify owned worker or remote-preregistered mode')
