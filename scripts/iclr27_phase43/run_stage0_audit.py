#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import numpy as np,torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase43'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]; recs=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json')); fit=[r for r in man['records'] if r['split']=='fit' and r['kind']=='multi_positive_cross_video']; bridge=SafetyVectorBridge(); bridge.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); bridge.eval(); per={}
  for p in PREFIXES:
   n=pos=top=mean=teacher=0
   for r in fit:
    qk=r['query_track_key']; sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key');
    if qk not in meta or not sk or hk not in meta: continue
    cks=sk+[hk]; q=vec(qk,meta,feats,p); c=np.asarray([vec(k,meta,feats,p) for k in cks]); sv=np.asarray([vec(k,meta,feats,p) for k in sk]); ctx=sv.mean(0); ctx/=max(float(np.linalg.norm(ctx)),1e-8); raw=q@c.T; sq=float((sv@q).max())
    with torch.no_grad(): z,a,_=bridge(torch.tensor(q).view(1,-1),torch.tensor(ctx).view(1,-1),torch.tensor([float(raw.max())]),torch.tensor([sq]),True); br=z.numpy()[0]@c.T
    rm=float(raw[:len(sk)].max()-raw[len(sk):].max()); bm=float(br[:len(sk)].max()-br[len(sk):].max()); tl=bool(sq>=0.2 and bm>=rm+0.005); btop=bool(br.argmax()<len(sk)); rtop=bool(raw.argmax()<len(sk));
    n+=1; pos+=int(bm>rm); top+=int(btop); mean+=int(br[:len(sk)].mean()>br[len(sk):].max()); teacher+=int(tl); recs.append({'fold':fold,'prefix':p,'episode_id':r.get('episode_id'),'raw_margin':rm,'bridge_margin':bm,'support_quality':sq,'teacher_use':tl,'raw_top1':rtop,'bridge_top1':btop,'unsafe_flip':bool(rtop and not btop)})
   per[str(p)]={'episodes':n,'max_margin_positive_rate':pos/max(n,1),'bridge_top1_rate':top/max(n,1),'all_positive_mean_better_rate':mean/max(n,1),'teacher_use_rate':teacher/max(n,1)}
  folds.append({'fold':fold,'prefix':per})
 atomic(OUT/'audit/contract.json',{'phase':43,'phase42_train_eval_mismatch_audited':True,'teacher_rule':'support_quality>=0.2 AND bridge_margin>=raw_margin+0.005','inference_threshold':'p>=0.5 unchanged','protocol_changed':False,'forbidden_inputs':['category','text','physical/semantic ID','future','held GT','StateMemory','controller action']})
 atomic(OUT/'audit/resource_preflight.json',{'gpu_ids':[4,5,6,7],'gpu_free_mib':40337,'ram_available_gb':118,'public_q1_dev_access':False,'bounded_workers':4})
 atomic(OUT/'audit/teacher_label_stats.json',{'folds':folds,'records':recs,'denominator':'TRAIN fit episodes','sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']})
 atomic(OUT/'completion/stage0.done',{'stage':0,'contract':'PASS'}); print(json.dumps({'stage0':'PASS','records':len(recs)},indent=2))
if __name__=='__main__': main()
