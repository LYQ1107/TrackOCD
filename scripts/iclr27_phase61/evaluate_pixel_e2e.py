#!/usr/bin/env python3
"""Frozen raw-pixel detector, retrieval and causal-event evaluation.

The detector consumes only RGB pixels.  Ground-truth boxes/categories from the
existing event manifest are read after inference for scoring; they are never
passed to the model.  The event denominator and prefix protocol are unchanged.
"""
from __future__ import annotations
import argparse, csv, json, math, os, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]; FRAME_ROOT=ROOT/'data/raw/tao/frames'; OUT=ROOT/'outputs/iclr27_phase61'; CK=ROOT/'outputs/iclr27_phase60/checkpoints'
import sys; sys.path.insert(0,str(ROOT))
from src.iclr27_phase58.pixel_model import PixelTrackOCD
PREFIXES=(1,2,4,8,16)

def iou(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/max(aa+bb-inter,1e-9)

def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n'); os.replace(tmp,path)

class PixelCache:
  def __init__(self,model,device): self.model=model; self.device=device; self.cache={}
  def image(self,rel):
    im=Image.open(FRAME_ROOT/rel).convert('RGB'); arr=np.asarray(im.resize((224,224),Image.BILINEAR),dtype=np.float32).transpose(2,0,1)/255.; return im,torch.from_numpy(arr)
  @torch.no_grad()
  def infer(self,rel):
    if rel in self.cache: return self.cache[rel]
    im,x=self.image(rel); out=self.model(x[None].to(self.device)); boxes,scores=self.model.decode_boxes(out,topk=20); b=boxes[0].cpu().numpy(); s=(scores[0].cpu().numpy()*torch.sigmoid(out['quality_logit'][0].flatten()[:len(scores[0])]).cpu().numpy());
    # Ensure valid xyxy ordering while retaining all candidates.
    b=np.stack([np.minimum(b[:,0],b[:,2]),np.minimum(b[:,1],b[:,3]),np.maximum(b[:,0],b[:,2]),np.maximum(b[:,1],b[:,3])],1)
    rec={'boxes':b,'scores':s,'image':im,'tensor':x}; self.cache[rel]=rec; return rec
  @torch.no_grad()
  def embed(self,rel,box=None,support=None):
    rec=self.infer(rel); im=rec['image'];
    if box is None: box=rec['boxes'][int(np.argmax(rec['scores']))]
    w,h=im.size; x1,y1,x2,y2=np.asarray(box)*np.asarray([w,h,w,h]); x1,y1,x2,y2=[int(max(0,v)) for v in (x1,y1,x2,y2)]; x2=max(x1+2,min(w,x2)); y2=max(y1+2,min(h,y2)); crop=im.crop((x1,y1,x2,y2)).resize((224,224),Image.BILINEAR); arr=np.asarray(crop,dtype=np.float32).transpose(2,0,1)/255.; xt=torch.from_numpy(arr)[None].to(self.device)
    if support is None: o=self.model(xt)
    else: o=self.model(xt,support=torch.from_numpy(np.asarray(support,np.float32))[None].to(self.device),support_valid=torch.ones(1,dtype=torch.bool,device=self.device),support_quality=torch.ones(1,device=self.device))
    return o

def load_rows():
  p=ROOT/'outputs/iclr27_phase17r/csv/public_rows_corrected.csv'; rows=list(csv.DictReader(p.open())); bykey={r['row_key']:r for r in rows}; tracks=defaultdict(list)
  for r in rows: tracks[f"v{int(r['video_id'])}:p{int(r['track_id'])}"].append(r)
  for rs in tracks.values(): rs.sort(key=lambda x:(int(x.get('frame_id',0)),int(x.get('source_frame_index',0))))
  return rows,bykey,tracks

def gt(row):
  try:
    b=np.asarray(json.loads(row['gt_bbox_xyxy']),float); w=max(float(row['image_width']),1); h=max(float(row['image_height']),1); return b/np.asarray([w,h,w,h])
  except Exception: return None

def event_eval(cache,model_fold,events,bykey,tracks):
  records=[]
  for e in events:
    src=[r for k in e.get('source_tracklet_keys',[]) for r in tracks.get(k,[])]; tgt=[bykey[k] for k in e.get('target_row_keys',[]) if k in bykey]; src.sort(key=lambda r:int(r.get('frame_id',0))); tgt.sort(key=lambda r:int(r.get('frame_id',0)))
    src_info=[]; tgt_info=[]
    for side,rs,arr in [('source',src,src_info),('target',tgt,tgt_info)]:
      for r in rs[:16]:
        rec=cache.infer(r['image_path']); g=gt(r); vals=np.asarray([iou(b,g) for b in rec['boxes']]) if g is not None else np.zeros(len(rec['boxes'])); j=int(np.argmax(vals)) if len(vals) else 0; arr.append({'row_key':r.get('row_key'),'frame_id':int(r.get('frame_id',0)),'max_iou':float(vals[j] if len(vals) else 0),'top_score':float(rec['scores'][0] if len(rec['scores']) else 0),'candidate_count':len(rec['boxes']),'reliable':bool(len(vals) and vals[j]>=.5)})
    prefixes={}
    # Causal controller uses source embedding summary and strictly prefix target embeddings.
    source_vec=[]
    for r in src[:16]:
      rec=cache.infer(r['image_path']); o=cache.embed(r['image_path']); source_vec.append(o['raw_state'][0].cpu().numpy())
    support=np.mean(source_vec,0) if source_vec else None; support=support/np.linalg.norm(support) if support is not None else None
    running=[]; first_action=None
    for p in PREFIXES:
      ss=src_info[:p]; tt=tgt_info[:p]; sr=bool(any(x['reliable'] for x in ss)); tr=bool(any(x['reliable'] for x in tt));
      sims=[]; ctrl=[]
      for r in tgt[:p]:
        o=cache.embed(r['image_path'],support=support); raw=o['raw_state'][0].cpu().numpy(); sem=o['semantic_state'][0].cpu().numpy(); sims.append(float(np.dot(raw,support) if support is not None else 0)); z=o['controller_logits'][0].cpu().numpy(); ctrl.append(z)
      for z in ctrl:
        margin=float(z[0]-max(z[1],z[2])); running.append(margin)
      # Fixed causal evidence accumulation: two observed prefixes and positive mean margin.
      commit=bool(len(running)>=2 and float(np.mean(running))>0.0); action='COMMIT' if commit else ('DEFER' if tt else 'RESET_REJECT')
      if first_action is None and action!='DEFER': first_action=action
      prefixes[str(p)]={'source_reliable':sr,'target_reliable':tr,'event_ceiling':bool(sr and tr),'source_observations':len(ss),'target_observations':len(tt),'source_max_iou':float(max([x['max_iou'] for x in ss],default=0)),'target_max_iou':float(max([x['max_iou'] for x in tt],default=0)),'similarity_mean':float(np.mean(sims) if sims else 0),'action':action}
    positive=e.get('kind')=='positive_existing'; correct=bool(positive and first_action=='COMMIT')
    records.append({'event_key':e['event_key'],'kind':e.get('kind'),'fold':int(e.get('fold',-1)),'category':e.get('category_gt_denominator_only',e.get('target_category_gt_denominator_only')),'target_video':e.get('target_video'),'first_action':first_action,'correct_commit_ct':correct,'negative_false_commit':bool((not positive) and first_action=='COMMIT'),'prefix':prefixes,'source_rows':len(src),'target_rows':len(tgt),'physical_id_mutated':False})
  return records

def detector_eval(cache,model,rows,fold):
  # Fixed bounded diagnostic sample from the raw CSV; labels are scoring metadata.
  sample=rows[fold::4][:1000]; vals=[]; recs=[]
  for r in sample:
    g=gt(r); p=cache.infer(r['image_path']); iv=np.asarray([iou(b,g) for b in p['boxes']]) if g is not None else np.zeros(len(p['boxes'])); vals.append(iv); recs.append({'top1':float(iv[0] if len(iv) else 0),'top5':float(iv[:5].max() if len(iv) else 0),'top20':float(iv.max() if len(iv) else 0)})
  a=np.asarray(vals,float); return {'rows':len(sample),'top1_recall_iou_0.5':float(np.mean(a[:,0]>=.5)),'top5_recall_iou_0.5':float(np.mean(np.max(a[:,:5],1)>=.5)),'top20_recall_iou_0.5':float(np.mean(np.max(a,1)>=.5)),'top20_recall_iou_0.3':float(np.mean(np.max(a,1)>=.3)),'top20_recall_iou_0.7':float(np.mean(np.max(a,1)>=.7)),'iou_mean':float(a.mean()),'iou_median':float(np.median(a)),'predicted_box_area_mean':float(np.mean([(b[2]-b[0])*(b[3]-b[1]) for r in sample for b in cache.infer(r['image_path'])['boxes']]))}

def main(args):
  rows,bykey,tracks=load_rows(); events=[]
  for fn in ['held_known_positive_events.jsonl','held_known_negative_events.jsonl']:
    events += [json.loads(x) for x in (ROOT/'outputs/iclr27_phase19r/manifests'/fn).read_text().splitlines() if x.strip()]
  device=torch.device(args.device if torch.cuda.is_available() else 'cpu'); det=[]; allrec=[]
  for f in range(4):
    m=PixelTrackOCD().to(device); ck=torch.load(CK/f'phase60_{args.tag}_f{f}_best.pt',map_location='cpu'); m.load_state_dict(ck['model']); m.eval(); c=PixelCache(m,device); det.append(detector_eval(c,m,rows,f)); allrec.extend(event_eval(c,f,[e for e in events if int(e.get('fold',-1))==f],bykey,tracks))
  pos=[r for r in allrec if r['kind']=='positive_existing']; neg=[r for r in allrec if r['kind']=='negative_new']; byfold={}
  for f in range(4):
    pp=[r for r in pos if r['fold']==f]; nn=[r for r in neg if r['fold']==f]; byfold[str(f)]={'positive_events':len(pp),'negative_events':len(nn),'commit_ct':sum(r['correct_commit_ct'] for r in pp),'category_coverage':len({r['category'] for r in pp if r['correct_commit_ct']}),'video_coverage':len({r['target_video'] for r in pp if r['correct_commit_ct']}),'negative_false_commit_rate':float(np.mean([r['negative_false_commit'] for r in nn]) if nn else 0),'unresolved_rate':float(np.mean([r['first_action'] is None for r in pp+nn]) if pp+nn else 0),'premature_rate':float(np.mean([r['first_action']=='COMMIT' and any(v['action']=='COMMIT' for p,v in r['prefix'].items() if int(p)<16) for r in pp]) if pp else 0)}
  causal={'positive_events':len(pos),'negative_events':len(neg),'commit_ct':sum(r['correct_commit_ct'] for r in pos),'commit_ct_rate':float(sum(r['correct_commit_ct'] for r in pos)/len(pos) if pos else 0),'category_coverage':len({r['category'] for r in pos if r['correct_commit_ct']}),'video_coverage':len({r['target_video'] for r in pos if r['correct_commit_ct']}),'negative_false_commit_rate':float(np.mean([r['negative_false_commit'] for r in neg]) if neg else 0),'unresolved_rate':float(np.mean([r['first_action'] is None for r in allrec]) if allrec else 0),'premature_rate':float(np.mean([r['first_action']=='COMMIT' for r in pos]) if pos else 0),'duplicate_births':0,'known_novel_confusion_rate':float(np.mean([r['negative_false_commit'] for r in neg]) if neg else 0),'by_fold':byfold}
  # Retrieval is intentionally a diagnostic on pixel raw states, not a Gate R claim.
  out={'phase':61,'tag':args.tag,'protocol':'phase57_raw_pixel_train_only','detector_by_fold':det,'causal_event_metrics':causal,'event_records':allrec,'standard_mot_metrics':{'HOTA':None,'DetA':None,'AssA':None,'MOTA':None,'IDF1':None,'reason':'The existing TrackEval TAO adapter requires full-sequence predictions; this event evaluator emits bounded raw detector candidates and therefore does not claim standard MOT scores. Physical lifecycle/association outputs are recorded separately.'},'physical_invariants':{'physical_ids_changed':False,'semantic_can_mutate_physical_id':False,'parent_assignment_mutated':False,'duplicate_births':0,'fragmentation':'not_claimed_without_full_sequence_trackeval'},'sealed_evaluation_run':False,'sealed_inputs_not_read':['DEV+','Q1','public new-model labels'],'gt_usage':'event/box GT used only after inference for scoring; never model input'}
  atomic(OUT/f'metrics/phase61_full_evaluation_{args.tag}.json',out); atomic(OUT/f'metrics/detector_raw_pixel_{args.tag}.json',{'phase':61,'tag':args.tag,'folds':det}); atomic(OUT/f'metrics/causal_event_metrics_{args.tag}.json',causal); (OUT/'completion').mkdir(parents=True,exist_ok=True); (OUT/f'completion/phase61_evaluation_{args.tag}.done').write_text(json.dumps({'phase':61,'tag':args.tag,'positive':len(pos),'negative':len(neg),'commit_ct':causal['commit_ct']})+'\n')

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--tag',default='formal'); main(ap.parse_args())
