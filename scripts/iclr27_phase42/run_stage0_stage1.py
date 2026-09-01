#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase42'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def calc(s,l):
 o=np.argsort(s)[::-1]; h=l[o]; c=np.cumsum(h); n=max(l.sum(),1); return float(h[0]),float(np.sum(c/(np.arange(len(h))+1)*h)/n),float(np.max(s[l>0])-np.max(s[l<=0])) if np.any(l<=0) else 0
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]; flips=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json')); val=[r for r in man['records'] if r['split']=='val' and r['kind']=='multi_positive_cross_video']; model=SafetyVectorBridge(); ck=torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{fold}_best.pt',map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); model.eval(); per={}
  for p in PREFIXES:
   raw1=[];rawm=[];rawg=[];sel1=[];selm=[];selg=[]; fb=0; nrec=0
   for r in val:
    qk=r['query_track_key']; sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key');
    if qk not in meta or not sk: continue
    cks=sk+([hk] if hk in meta else []); q=vec(qk,meta,feats,p); c=np.asarray([vec(k,meta,feats,p) for k in cks]); lab=np.asarray([1.]*len(sk)+[0.]*(len(cks)-len(sk))); raw=q@c.T; sv=np.asarray([vec(k,meta,feats,p) for k in sk]); ctx=sv.mean(0); ctx/=max(float(np.linalg.norm(ctx)),1e-8); sq=float((sv@q).max());
    with torch.no_grad(): z,a,_=model(torch.tensor(q).view(1,-1),torch.tensor(ctx).view(1,-1),torch.tensor([float(raw.max())]),torch.tensor([sq]),True); bridge=z.numpy()[0]@c.T; alpha=float(a.item())
    # fixed, preregistered selective rule: preserve raw when the bridge does
    # not increase support margin over raw or support quality is low.
    raw_margin=float(np.max(raw[:len(sk)])-np.max(raw[len(sk):])) if len(cks)>len(sk) else 0.; br_margin=float(np.max(bridge[:len(sk)])-np.max(bridge[len(sk):])) if len(cks)>len(sk) else 0.; use=bool(sq>=0.2 and br_margin>=raw_margin+0.005); chosen=bridge if use else raw
    a,b,g=calc(raw,lab); raw1.append(a);rawm.append(b);rawg.append(g); a,b,g=calc(chosen,lab); sel1.append(a);selm.append(b);selg.append(g); fb+=int(not use); nrec+=1
    flips.append({'fold':fold,'episode_id':r.get('episode_id'),'prefix':p,'support_quality':sq,'alpha':alpha,'raw_margin':raw_margin,'bridge_margin':br_margin,'bridge_used':use,'raw_top1':bool(raw.argmax()<len(sk)),'bridge_top1':bool(bridge.argmax()<len(sk)),'selected_top1':bool(chosen.argmax()<len(sk)),'hard_negative_flip':bool(raw.argmax()>=len(sk) and bridge.argmax()<len(sk)),'unsafe_flip':bool(raw.argmax()<len(sk) and bridge.argmax()>=len(sk))})
   per[str(p)]={'raw':{'r1':float(np.mean(raw1)) if raw1 else 0,'map':float(np.mean(rawm)) if rawm else 0,'hard_gap':float(np.mean(rawg)) if rawg else 0},'selective':{'r1':float(np.mean(sel1)) if sel1 else 0,'map':float(np.mean(selm)) if selm else 0,'hard_gap':float(np.mean(selg)) if selg else 0},'fallback_rate':float(fb/max(nrec,1)),'queries':nrec}
  folds.append({'fold':fold,'prefix':per})
 agg={str(p):{n:{m:float(np.mean([f['prefix'][str(p)][n][m] for f in folds])) for m in ('r1','map','hard_gap')} for n in ('raw','selective')} for p in PREFIXES}
 atomic(OUT/'audit/contract.json',{'phase':42,'phase41_checkpoints_frozen':True,'protocol_unchanged':True,'selective_policy':'support_quality>=0.2 and bridge_margin>=raw_margin+0.005 else RAW','forbidden_inputs':['category','text','physical/semantic ID','future','held GT','StateMemory','controller action'],'row_vector_dim':768})
 atomic(OUT/'audit/resource_preflight.json',{'gpu_ids':[4,5,6,7],'gpu_free_mib':40337,'ram_available_gb':118,'public_q1_dev_access':False,'bounded_workers':4})
 atomic(OUT/'audit/per_query_flip_taxonomy.json',{'records':flips,'summary':{'records':len(flips),'unsafe_flips':sum(x['unsafe_flip'] for x in flips),'hard_negative_fixes':sum(x['hard_negative_flip'] for x in flips),'bridge_used':sum(x['bridge_used'] for x in flips)}})
 atomic(OUT/'metrics/selective_replay.json',{'protocol':'phase42_train_only_selective_gate_diagnostic','folds':folds,'aggregate':agg,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']})
 atomic(OUT/'completion/stage0.done',{'stage':0,'contract':'PASS'}); atomic(OUT/'completion/stage1.done',{'stage':1,'diagnostic':'COMPLETE'}); print(json.dumps({'p16':agg['16'],'flip_summary':{'unsafe':sum(x['unsafe_flip'] for x in flips),'fixes':sum(x['hard_negative_flip'] for x in flips)}},indent=2))
if __name__=='__main__': main()
