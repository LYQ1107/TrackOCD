#!/usr/bin/env python3
"""Replay a frozen association checkpoint on the Q0 event stream.

This evaluator is post-inference only: event GT is joined after the learned
stream is written to compute observability.  It never supplies labels to the
runtime and keeps the 76-event denominator fixed.
"""
from __future__ import annotations
import argparse, collections, datetime, hashlib, json, os, statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NATIVE=Path('/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl')
EVENT_OBS=Path('/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl')
EVENT_ROWS=ROOT/'data/iclr27_phase19r/sources/public_rows_corrected.csv'

def iou(a,b):
    if a is None or b is None:return 0.0
    ax0,ay0,ax1,ay1=a; bx0,by0,bx1,by1=b; ix0=max(ax0,bx0); iy0=max(ay0,by0); ix1=min(ax1,bx1); iy1=min(ay1,by1); inter=max(0,ix1-ix0)*max(0,iy1-iy0); aa=max(0,ax1-ax0)*max(0,ay1-ay0); ab=max(0,bx1-bx0)*max(0,by1-by0); return inter/(aa+ab-inter) if aa+ab-inter>0 else 0.0

def load_csv():
    import csv,ast
    rows={}
    with EVENT_ROWS.open() as f:
        for r in csv.DictReader(f):
            r['video_id']=int(r['video_id']); r['image_id']=int(r['image_id']); r['frame_id']=int(r['frame_id']); r['bbox_xyxy']=ast.literal_eval(r['bbox_xyxy']);
            try:r['gt_bbox_xyxy']=ast.literal_eval(r['gt_bbox_xyxy']) if r['gt_bbox_xyxy'] and r['gt_bbox_xyxy']!='nan' else None
            except Exception:r['gt_bbox_xyxy']=None
            rows[r['row_key']]=r
    return rows

def load_event_records():
    return [json.loads(x) for x in EVENT_OBS.read_text().splitlines() if x.strip()]

def run_stream(ckpt:Path, device:str, max_videos=None, use_motion: bool = False, max_miss: int = 8, use_appearance: bool = False, geometry: bool = False, geometry_conservative: bool = False, geometry_history: bool = False):
    import torch
    from src.iclr27_phase81p.association import AssociationTransformer, CausalAssociationRuntime, CausalGeometryRuntime, crop_descriptor
    if geometry:
        runtime=CausalGeometryRuntime(max_miss=int(max_miss), max_tracks=512, conservative=geometry_conservative, history=geometry_history)
    else:
        state=torch.load(str(ckpt),map_location='cpu'); model=AssociationTransformer(); model.load_state_dict(state.get('model',state)); runtime=CausalAssociationRuntime(model,device=device,max_miss=int(max_miss),match_margin=0.0,max_tracks=256,use_motion=use_motion)
    frames=collections.defaultdict(lambda:collections.defaultdict(list)); allowed=None
    if max_videos is not None:
        vids=sorted({int(x['source_video']) for x in load_event_records()}|{int(x['target_video']) for x in load_event_records()}); allowed=set(vids[:max_videos])
    with NATIVE.open() as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line); v=int(x['video_id']);
            if allowed is not None and v not in allowed:continue
            if x.get('bbox_xyxy') is not None: frames[v][(int(x.get('frame_id',0)),int(x.get('image_id',-1)))].append(x)
    out=[]; 
    for v in sorted(frames):
        runtime.tracks=[]; runtime.next_id=0
        for (frame,image),dets in sorted(frames[v].items()):
            # Appearance is optional in this first physical route; geometry,
            # score and causal history remain fully available.
            for d in dets:
                if use_appearance:
                    frame_path=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames')/str(d.get('file_path',''))
                    d['appearance']=crop_descriptor(str(frame_path),d['bbox_xyxy']) if frame_path.is_file() else np.zeros(8,np.float32)
                else:
                    d['appearance']=np.zeros(8,np.float32)
            out.extend(runtime.step(dets,frame))
    return out

def event_metrics(out, events):
    by_key=collections.defaultdict(list)
    for r in out: by_key[(int(r['video_id']),int(r['image_id']))].append(r)
    csv_rows=load_csv(); p16=[e for e in events if e.get('prefix')==16 and e.get('polarity')=='positive']; result=[]
    for e in p16:
        sides={}
        for side in ('source','target'):
            details=e.get(side+'_row_details',[]); per_track=collections.defaultdict(list); row_reliable=0
            for det in details:
                key=det.get('row_key'); gt=csv_rows.get(key,{}).get('gt_bbox_xyxy'); cand=by_key.get((int(det.get('video_id',-1)),int(det.get('image_id',-1))),[])
                best=max(cand,key=lambda x:iou(x.get('bbox_xyxy'),gt),default=None); score=iou(best.get('bbox_xyxy'),gt) if best else 0.0
                if score>=0.5: row_reliable+=1
                if best: per_track[int(best['physical_track_id'])].append(score)
            track_stats=[{'track_id':tid,'coverage':len(vals)/max(1,len(details)),'mean_iou':float(statistics.mean(vals)),'max_iou':max(vals)} for tid,vals in per_track.items()]
            best_track=max(track_stats,key=lambda x:(x['coverage']*x['mean_iou'],x['coverage']),default={'track_id':None,'coverage':0.0,'mean_iou':0.0,'max_iou':0.0})
            sides[side]={'row_reliable_count':row_reliable,'rows':len(details),'track_stats':track_stats,'best_track':best_track,'reliable':bool(row_reliable>0 and best_track['mean_iou']>=0.5)}
        result.append({'event_key':e['event_key'],'fold':e['fold'],'source':sides['source'],'target':sides['target'],'both_reliable':sides['source']['reliable'] and sides['target']['reliable']})
    agg={'event_count':len(result),'both_reliable':sum(x['both_reliable'] for x in result),'source_reliable':sum(x['source']['reliable'] for x in result),'target_reliable':sum(x['target']['reliable'] for x in result),'by_fold':{}}
    for f in range(4):
        rr=[x for x in result if x['fold']==f]; agg['by_fold'][str(f)]={'events':len(rr),'both_reliable':sum(x['both_reliable'] for x in rr),'source_reliable':sum(x['source']['reliable'] for x in rr),'target_reliable':sum(x['target']['reliable'] for x in rr)}
    return {'aggregate':agg,'events':result}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=False,default=''); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--tag',default='formal'); ap.add_argument('--max-videos',type=int); ap.add_argument('--motion',action='store_true'); ap.add_argument('--appearance',action='store_true'); ap.add_argument('--geometry',action='store_true'); ap.add_argument('--geometry-conservative',action='store_true'); ap.add_argument('--geometry-history',action='store_true'); ap.add_argument('--max-miss',type=int,default=8); args=ap.parse_args()
    import torch
    ck=Path(args.checkpoint) if args.checkpoint else Path('/dev/null'); events=load_event_records(); stream=run_stream(ck,args.device,args.max_videos,use_motion=args.motion,max_miss=args.max_miss,use_appearance=args.appearance,geometry=args.geometry,geometry_conservative=args.geometry_conservative,geometry_history=args.geometry_history); out=event_metrics(stream,events); result={'schema_version':'phase81p.replay_metrics.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint':str(ck.resolve()) if not args.geometry else None,'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest() if not args.geometry else None,'event_observability_input':str(EVENT_OBS),'event_observability_sha256':hashlib.sha256(EVENT_OBS.read_bytes()).hexdigest(),'aggregate':out['aggregate'],'events':out['events'],'protocol':{'positive_denominator':76,'negative_denominator':76,'prefixes':[1,2,4,8,16],'labels_joined_before_inference':False,'future_rows_or_tracks':False,'ids_as_model_input':False,'causal_motion_prediction':bool(args.motion),'causal_appearance_descriptor':bool(args.appearance),'geometry_only':bool(args.geometry),'geometry_conservative':bool(args.geometry_conservative),'geometry_history':bool(args.geometry_history),'max_miss':int(args.max_miss)}}
    path=ROOT/f'outputs/iclr27_phase81p/metrics/replay_{args.tag}.json'; path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name('.'+path.name+'.tmp'); tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path); print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__': main()
