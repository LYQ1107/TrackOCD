#!/usr/bin/env python3
"""Build TRAIN-only causal association pair/listwise shards.

Only annotation geometry and RGB crop statistics become model inputs.  GT
track/category values are retained in the shard solely as supervision metadata
and are never serialized as feature columns.
"""
from __future__ import annotations
import collections, datetime, hashlib, json, os, random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from src.iclr27_phase81p.association import crop_descriptor, pair_features
TRAIN_JSON = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json")
EVENT_MANIFEST = ROOT / "outputs/iclr27_phase74s/manifests/model_events_v2.jsonl"
OUT = ROOT / "outputs/iclr27_phase81p/manifests"
DATA = Path("/data2/usr_for_deadline/trackocd_phase81p/data")
SEED = 8101

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name('.'+path.name+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def event_videos():
    vids=set()
    for line in EVENT_MANIFEST.read_text().splitlines():
        if line.strip():
            x=json.loads(line); vids.update([int(x['source_video']),int(x['target_video'])])
    return vids

def make_descriptor_cache(images, anns, use_appearance=False):
    by_img=collections.defaultdict(list)
    for a in anns: by_img[int(a['image_id'])].append(a)
    cache={}; missing=0
    for iid, rows in by_img.items():
        im=images.get(iid); rel=im.get('file_name') if im else None
        path=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames')/str(rel) if rel else None
        for a in rows:
            b=a['bbox']; box=[float(b[0]),float(b[1]),float(b[0]+b[2]),float(b[1]+b[3])]
            desc=crop_descriptor(str(path),box) if use_appearance and path else np.zeros(8,np.float32)
            if use_appearance and (not path or not path.is_file()): missing+=1
            cache[int(a['id'])]=(box,desc)
    return cache,missing

def build_fold(fold, videos, images, anns, ann_cache, held_categories, data_dir, use_motion=False):
    # Deterministic video/category-disjoint split; event videos are excluded.
    val_vids={v for v in videos if v % 4 == fold}
    fit_vids=set(videos)-val_vids
    fit=[]; val=[]
    by_video=collections.defaultdict(list)
    for a in anns:
        if int(a.get('iscrowd',0)): continue
        iid=int(a['image_id']); vid=int(images[iid]['video_id']); cat=int(a['category_id'])
        if vid in fit_vids and cat not in held_categories: fit.append(a)
        if vid in val_vids and cat in held_categories: val.append(a)
    def examples(rows, limit=50000):
        by_vf=collections.defaultdict(list); image_frame={int(i):int(im.get('frame_index',0)) for i,im in images.items()}
        for a in rows: by_vf[(int(images[int(a['image_id'])]['video_id']),int(a['image_id']))].append(a)
        videos_frames=collections.defaultdict(list)
        for (v,iid) in by_vf: videos_frames[v].append(iid)
        X=[]; y=[]; gap=[]; pos_count=0; neg_count=0; new_count=0
        for vid, frame_ids in videos_frames.items():
            frame_ids.sort(key=lambda iid:(image_frame[iid],iid)); history={}
            for pos, iid in enumerate(frame_ids):
                cur=by_vf[(vid,iid)]
                frame=int(image_frame[iid]); candidates=list(history.values())
                # A causal tracker cannot retain arbitrarily old candidates;
                # pruning to the registered eight-frame memory also prevents
                # quadratic work on long TAO videos.
                history={k:v for k,v in history.items() if pos-int(v['pos']) <= 8}
                candidates=list(history.values())
                # Keep up to eight recent alternatives plus the true trajectory.
                for a in cur:
                    gt=int(a['track_id']); box,desc=ann_cache[int(a['id'])]
                    prev=[h for h in candidates if h['track_id']==gt]
                    others=[h for h in candidates if h['track_id']!=gt]
                    others.sort(key=lambda h: abs(float(h['box'][0]-box[0]))+abs(float(h['box'][1]-box[1])))
                    chosen=(prev[:1]+others[:8])
                    feats=[]; target=9  # explicit NEW class unless a causal positive exists
                    for j,h in enumerate(chosen):
                        d={'bbox_xyxy':box,'appearance':desc,'frame_id':frame,'base_score':1.0}
                        t={'last_bbox':h['box'],'appearance_ema':h['appearance'],'last_frame':h['frame'],'velocity':h.get('velocity',np.zeros(4,np.float32)),'age':h['age'],'miss_count':max(0,pos-int(h['pos'])),'score_ema':1.0,'association_ema':0.0,'hit_count':h['age']}
                        feats.append(pair_features(d,t,use_motion=use_motion))
                        if int(h['track_id'])==gt: target=j
                    # A first observation has no active history.  Retain a
                    # zero-context sample so the NEW head receives an actual
                    # TRAIN target; the zero vector is a structural marker,
                    # not an identity/category feature.
                    if not feats:
                        arr=np.zeros((9,16),np.float32)
                        X.append(arr); y.append(9); gap.append(0)
                        new_count += 1
                        if len(X)>=limit: break
                        continue
                    # Fixed-width candidates: zero-pad; target index is preserved.
                    k=min(9,len(feats)); arr=np.zeros((9,16),np.float32); arr[:k]=np.asarray(feats[:k],np.float32)
                    X.append(arr); y.append(target if target < k else 9); gap.append(min(8,max(0,pos-(chosen[0]['pos'] if chosen else pos))))
                    pos_count += int(target < k); new_count += int(target >= k); neg_count += max(0,k-1)
                    if len(X)>=limit: break
                # Update causal latest history after scoring this frame.
                for a in cur:
                    box,desc=ann_cache[int(a['id'])]; old=history.get(int(a['track_id']))
                    if old is None:
                        velocity=np.zeros(4,np.float32); age=1
                    else:
                        velocity=0.8*np.asarray(old.get('velocity',np.zeros(4,np.float32)),np.float32)+0.2*(np.asarray(box,np.float32)-np.asarray(old['box'],np.float32))/float(max(1,frame-int(old['frame']))); age=int(old['age'])+1
                    history[int(a['track_id'])]={'track_id':int(a['track_id']),'box':box,'appearance':desc,'velocity':velocity,'frame':frame,'pos':pos,'age':age}
                if len(X)>=limit: break
            if len(X)>=limit: break
        if not X: return np.zeros((0,9,16),np.float32),np.zeros((0,),np.int64),{'examples':0,'positive':0,'new':0,'hard_negatives':0}
        return np.stack(X),np.asarray(y,np.int64),{'examples':len(X),'positive':pos_count,'new':new_count,'hard_negatives':neg_count}
    fit_x,fit_y,fit_stats=examples(fit); val_x,val_y,val_stats=examples(val)
    data_dir.mkdir(parents=True,exist_ok=True)
    fp=data_dir/f'fold{fold}.npz'; vp=data_dir/f'fold{fold}_val.npz'; np.savez_compressed(fp,x=fit_x,y=fit_y); np.savez_compressed(vp,x=val_x,y=val_y)
    return {'fold':fold,'fit_videos':sorted(fit_vids),'val_videos':sorted(val_vids),'held_categories':sorted(held_categories),'fit_categories_disjoint':True,'fit':fit_stats,'val':val_stats,'fit_path':str(fp),'val_path':str(vp)}

def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--appearance',action='store_true',help='decode RGB crops (slower; omitted for default geometry smoke)'); parser.add_argument('--motion',action='store_true',help='use causal velocity prediction in the fixed-width pair vector'); parser.add_argument('--route-tag',default='baseline',help='isolated manifest/data namespace'); args=parser.parse_args()
    random.seed(SEED); np.random.seed(SEED)
    ann=json.loads(TRAIN_JSON.read_text()); images={int(x['id']):x for x in ann['images']}; anns=ann['annotations']; vids=sorted({int(x['video_id']) for x in ann['images']} - event_videos()); categories=sorted({int(x['category_id']) for x in anns})
    cache,missing=make_descriptor_cache(images,anns,use_appearance=args.appearance)
    folds=[]; data_dir=DATA/args.route_tag
    for f in range(4): folds.append(build_fold(f,vids,images,anns,cache,{c for c in categories if c%4==f},data_dir,use_motion=args.motion))
    result={'schema_version':'phase81p.train_manifest.v3','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':SEED,'route_tag':args.route_tag,'train_annotations':str(TRAIN_JSON),'train_annotations_sha256':hashlib.sha256(TRAIN_JSON.read_bytes()).hexdigest(),'excluded_event_videos':sorted(event_videos()),'video_count_after_exclusion':len(vids),'category_count':len(categories),'descriptor':'8-D RGB crop mean/std when --appearance is enabled; otherwise deterministic zero appearance (geometry/score-only baseline)','appearance_enabled':bool(args.appearance),'motion_features':bool(args.motion),'missing_image_annotations':missing,'folds':folds,'data_dir':str(data_dir),'inference_tensor_forbidden':['track_id','category_id','physical_id','semantic_id','future','held_gt']}
    manifest_dir=OUT/args.route_tag; manifest_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(manifest_dir/'train_manifest.json',result); atomic_json(manifest_dir/'supervision_inventory.json',result); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
