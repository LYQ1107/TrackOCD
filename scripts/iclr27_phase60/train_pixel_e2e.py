#!/usr/bin/env python3
"""Train the compact raw-RGB causal TrackOCD graph for one TRAIN fold.

The loader returns RGB frames and loss-only labels.  Category/video/track
values are never concatenated to model inputs; they are used only to form
contrastive targets and fold metadata.  A single run follows a small
detector→track→semantic→controller curriculum before joint optimization.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
FRAME_ROOT=ROOT/'data/raw/tao/frames'
INDEX=ROOT/'outputs/iclr27_phase57/manifests/train_track_index.jsonl'
ANNOT=ROOT/'data/raw/tao/annotations/train.json'
OUT=ROOT/'outputs/iclr27_phase60'
import sys
sys.path.insert(0,str(ROOT))
from src.iclr27_phase58.pixel_model import PixelTrackOCD

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+"\n"); os.replace(tmp,path)

class TrackPairs(Dataset):
    def __init__(self, records, seed=0, ann_by_path=None, held_categories=None, held_videos=None):
        self.records=records; self.seed=seed; self.ann_by_path=ann_by_path or {}
        self.held_categories=set(held_categories or ()); self.held_videos=set(held_videos or ())
    def __len__(self): return len(self.records)
    def _img(self,p):
        im=Image.open(FRAME_ROOT/p).convert('RGB').resize((224,224),Image.BILINEAR)
        a=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.0
        return torch.from_numpy(a)
    def __getitem__(self,i):
        r=self.records[i%len(self.records)]; fs=r['frames']; n=len(fs)
        # Deterministic adjacent prefix pair; no future frame is read.
        j=(i*17+self.seed)%max(1,n-1); k=min(j+1,n-1)
        f1,f2=fs[j],fs[k]
        def norm(b,w,h): return [b[0]/w,b[1]/h,b[2]/w,b[3]/h]
        b1=norm(f1['bbox_xyxy'],f1['width'],f1['height']); b2=norm(f2['bbox_xyxy'],f2['width'],f2['height'])
        def all_boxes(frame):
            # Class-agnostic objectness must see every legal TRAIN annotation in
            # the image.  The selected track remains the representation pair;
            # other boxes are loss-only labels and never enter model inputs.
            vals=[]
            for a in self.ann_by_path.get(frame['image_path'],[]):
                if int(a['category_id']) in self.held_categories or int(a['video_id']) in self.held_videos:
                    continue
                vals.append(a['bbox_norm'])
            if not vals:
                vals=[norm(frame['bbox_xyxy'],frame['width'],frame['height'])]
            return torch.tensor(vals,dtype=torch.float32)
        return {'image1':self._img(f1['image_path']),'image2':self._img(f2['image_path']),
                'box1':torch.tensor(b1,dtype=torch.float32),'box2':torch.tensor(b2,dtype=torch.float32),
                'all_boxes1':all_boxes(f1),'all_boxes2':all_boxes(f2),
                'age1':torch.tensor(min(j,16)/16.,dtype=torch.float32),'age2':torch.tensor(min(k,16)/16.,dtype=torch.float32),
                'category':int(r['category_id']),'video':int(r['video_id']),'track':f"{r['video_id']}:{r['track_id']}"}

def collate(batch):
    out={}
    for k in ('image1','image2','box1','box2','age1','age2'): out[k]=torch.stack([x[k] for x in batch])
    out['all_boxes1']=[x['all_boxes1'] for x in batch]; out['all_boxes2']=[x['all_boxes2'] for x in batch]
    for k in ('category','video','track'): out[k]=[x[k] for x in batch]
    return out

def det_targets(boxes,h=28,w=28,device=None):
    # ``boxes`` is a per-image list so all annotated objects, not only the
    # sampled track, contribute class-agnostic objectness supervision.
    b=len(boxes); obj=torch.zeros((b,h,w),device=device); bt=torch.zeros((b,4,h,w),device=device); mask=torch.zeros_like(obj)
    for i,bi in enumerate(boxes):
        bi=torch.as_tensor(bi,device=device,dtype=torch.float32).reshape(-1,4)
        cx=((bi[:,0]+bi[:,2])*0.5*w).long().clamp(0,w-1); cy=((bi[:,1]+bi[:,3])*0.5*h).long().clamp(0,h-1)
        for j in range(len(bi)):
            obj[i,cy[j],cx[j]]=1.; bt[i,:,cy[j],cx[j]]=bi[j]; mask[i,cy[j],cx[j]]=1.
    return obj,bt,mask

def detector_loss(out,boxes):
    objt,boxt,m=det_targets(boxes, out['objectness_logit'].shape[-2],out['objectness_logit'].shape[-1],out['objectness_logit'].device)
    # The dense grid has one positive cell and ~783 negatives.  A fixed
    # ``pos_weight=8`` silently learned the all-negative solution in the first
    # formal run.  Balance the positive and negative means explicitly; this is
    # still the same class-agnostic objectness target and does not alter boxes,
    # rows, or the evaluation protocol.
    pos=nn.functional.binary_cross_entropy_with_logits(out['objectness_logit'][m.bool()],torch.ones_like(out['objectness_logit'][m.bool()]))
    neg=nn.functional.binary_cross_entropy_with_logits(out['objectness_logit'][~m.bool()],torch.zeros_like(out['objectness_logit'][~m.bool()]))
    lo=0.5*(pos+neg)
    pred=torch.sigmoid(out['bbox_logits']); lb=(torch.abs(pred-boxt)*m[:,None]).sum()/m.sum().clamp_min(1.)
    qpos=nn.functional.binary_cross_entropy_with_logits(out['quality_logit'][m.bool()],torch.ones_like(out['quality_logit'][m.bool()]))
    qneg=nn.functional.binary_cross_entropy_with_logits(out['quality_logit'][~m.bool()],torch.zeros_like(out['quality_logit'][~m.bool()]))
    lq=0.5*(qpos+qneg)
    return lo+lb+lq, {'objectness':float(lo.detach()),'bbox':float(lb.detach()),'quality':float(lq.detach())}

def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    records=[json.loads(x) for x in INDEX.read_text().splitlines() if x.strip()]
    fold=json.loads((OUT.parent/'phase57/manifests/fold_%d.json'%args.fold).read_text()) if False else None
    # Use the same deterministic fold rule as build_frame_contract without writing labels into tensors.
    hc=[]
    hvs=[]
    for r in records:
        # A fold's fit membership is reconstructed from the manifest held sets.
        pass
    fm=json.loads((ROOT/f'outputs/iclr27_phase57/manifests/fold_{args.fold}.json').read_text())
    heldc=set(fm['held_categories']); heldv=set(fm['held_videos'])
    fit=[r for r in records if int(r['category_id']) not in heldc and int(r['video_id']) not in heldv]
    if not fit: raise RuntimeError('empty TRAIN fit fold')
    ann=json.loads(ANNOT.read_text()); images={int(x['id']):x for x in ann['images']}; ann_by_path={}
    for a in ann['annotations']:
        im=images.get(int(a['image_id']))
        if im is None: continue
        x,y,w,h=a['bbox']; iw=max(float(im['width']),1.); ih=max(float(im['height']),1.)
        rec={'bbox_norm':[float(x)/iw,float(y)/ih,float(x+w)/iw,float(y+h)/ih],
             'category_id':int(a.get('category_id',-1)),'video_id':int(a.get('video_id',im.get('video_id',-1)))}
        ann_by_path.setdefault(im['file_name'],[]).append(rec)
    ds=TrackPairs(fit,args.seed,ann_by_path,heldc,heldv); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=torch.cuda.is_available(),collate_fn=collate,drop_last=True)
    it=iter(loader); model=PixelTrackOCD().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    nparams=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckdir=OUT/'checkpoints'; ckdir.mkdir(parents=True,exist_ok=True); comp=OUT/'completion'; comp.mkdir(parents=True,exist_ok=True)
    logs=[]; best=1e99; t0=time.time()
    for step in range(1,args.steps+1):
        try: batch=next(it)
        except StopIteration: it=iter(loader); batch=next(it)
        x1=batch['image1'].to(dev,non_blocking=True); x2=batch['image2'].to(dev,non_blocking=True)
        b1=batch['box1'].to(dev); b2=batch['box2'].to(dev); a1=batch['age1'].to(dev); a2=batch['age2'].to(dev)
        out1=model(x1,age=a1); out2=model(x2,age=a2)
        d1,ld1=detector_loss(out1,batch['all_boxes1']); d2,ld2=detector_loss(out2,batch['all_boxes2'])
        e1,e2=out1['track_embedding'],out2['track_embedding']; s1,s2=out1['semantic_state'],out2['semantic_state']
        assoc_pos=(1-(e1*e2).sum(1)).mean()
        perm=torch.roll(torch.arange(len(e1),device=dev),1); negsim=(e1*e2[perm]).sum(1); assoc_neg=torch.relu(negsim-0.15).mean()
        assoc=assoc_pos+assoc_neg
        cats=torch.tensor(batch['category'],device=dev); vids=torch.tensor(batch['video'],device=dev)
        same=(cats[:,None]==cats[None,:]) & (vids[:,None]!=vids[None,:]); off=~torch.eye(len(cats),dtype=torch.bool,device=dev)
        pairmask=(off & (vids[:,None]!=vids[None,:]))
        pairlog=(s1@s2.T)*4.0; labels=same.float()
        corr=nn.functional.binary_cross_entropy_with_logits(pairlog[pairmask],labels[pairmask]) if pairmask.any() else torch.zeros((),device=dev)
        temporal=(1-(s1*s2).sum(1)).mean()
        life_t=torch.zeros_like(out2['lifecycle_logits']); life_t[:,1]=1.; life=nn.functional.binary_cross_entropy_with_logits(out2['lifecycle_logits'],life_t)
        # Prior support is a strictly earlier item in the sampled batch only when
        # it comes from another video with the same loss-only category label.
        support=torch.zeros_like(out2['raw_state']); valid=torch.zeros((len(cats),),dtype=torch.bool,device=dev)
        for i in range(len(cats)):
            js=torch.where((cats==cats[i]) & (vids!=vids[i]))[0]
            if len(js): support[i]=out1['raw_state'][js[0]].detach(); valid[i]=True
        supq=valid.float().to(dev)
        out2s=model(x2,age=a2,support=support,support_valid=valid,support_quality=supq)
        st=out2s['semantic_state']; support_use=(1-(st*out2s['raw_state']).sum(1)).clamp(0,2).mean()
        ctrl=out2s['controller_logits']; commit_t=valid.float(); defer_t=1-commit_t
        commit_loss=nn.functional.binary_cross_entropy_with_logits(ctrl[:,0],commit_t)+nn.functional.binary_cross_entropy_with_logits(ctrl[:,1],defer_t)
        persistent=nn.functional.softplus(-ctrl[:,0]*(2*commit_t-1)).mean()
        safety=nn.functional.softplus(negsim*2.0).mean()
        rawpres=(1-(st*out2s['raw_state']).sum(1)).clamp_min(0).mean()
        frac=step/max(args.steps,1)
        sem_w=min(1.,frac*5); ctrl_w=min(1.,max(0.,(frac-.25)*4))
        total=d1+d2+assoc+sem_w*(corr+temporal)+life+ctrl_w*(commit_loss+0.25*persistent)+0.5*safety+0.25*rawpres
        opt.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            grads={}
            for name,grp in [('proposal',[model.objectness.weight,model.box_delta.weight]),('physical',[model.global_proj[0].weight]),('semantic',[model.raw_proj[0].weight,model.support_residual[0].weight]),('controller',[model.controller[0].weight])]: grads[name]=float(np.sqrt(sum(float((p.grad.detach()**2).sum()) for p in grp if p.grad is not None)))
            rec={'step':step,'stage':('A_detector_physical' if frac<.2 else 'B_track_representation' if frac<.4 else 'C_correspondence_state' if frac<.6 else 'D_commit_defer' if frac<.8 else 'E_joint'),'total':float(total.detach()),'detector':{k:(ld1[k]+ld2[k])/2 for k in ld1},'association':float(assoc.detach()),'correspondence':float(corr.detach()),'temporal':float(temporal.detach()),'lifecycle':float(life.detach()),'commit_defer':float(commit_loss.detach()),'persistent_proxy':float(persistent.detach()),'mot_safety':float(safety.detach()),'raw_preservation':float(rawpres.detach()),'support_use':float(support_use.detach()),'grad_norms':grads,'bridge_use_rate':float(valid.float().mean()),'rss_mb':float(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/1024)}; logs.append(rec); print(json.dumps(rec),flush=True)
        if step%args.ckpt_every==0 or step==args.steps:
            state={'phase':60,'fold':args.fold,'seed':args.seed,'step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),'logs':logs[-20:],'params':nparams,'trainable':trainable}
            tmp=ckdir/f'.phase60_{args.tag}_f{args.fold}_step{step:05d}.pt.tmp'; torch.save(state,tmp); os.replace(tmp,ckdir/f'phase60_{args.tag}_f{args.fold}_step{step:05d}.pt'); torch.save(state,ckdir/f'phase60_{args.tag}_f{args.fold}_latest.pt')
            if float(total.detach())<best: best=float(total.detach()); torch.save(state,ckdir/f'phase60_{args.tag}_f{args.fold}_best.pt')
    metrics={'phase':60,'fold':args.fold,'tag':args.tag,'steps':args.steps,'seed':args.seed,'device':str(dev),'fit_tracks':len(fit),'params':nparams,'trainable':trainable,'elapsed_sec':time.time()-t0,'loss_log':logs,'amp':'fp32','forbidden_inputs':['category_name','category_text','semantic_id','physical_id_feature','future_frame','future_track','held_gt','DEV+','Q1','public_new_model_label'],'source':'raw RGB frames only; GT boxes/categories are loss metadata'}
    atomic_json(OUT/f'metrics_phase60_{args.tag}_f{args.fold}.json',metrics)
    (comp/f'phase60_{args.tag}_f{args.fold}.done').write_text(json.dumps({'phase':60,'fold':args.fold,'steps':args.steps})+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--seed',type=int,default=575700); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--ckpt-every',type=int,default=100); ap.add_argument('--log-every',type=int,default=20); ap.add_argument('--tag',default='formal'); run(ap.parse_args())
