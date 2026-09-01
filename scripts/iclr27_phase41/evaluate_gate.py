#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import numpy as np,torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase41'; TAG=os.environ.get('PHASE41_TAG','gate_formal'); PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def calc(s,l):
 o=np.argsort(s)[::-1]; h=l[o]; c=np.cumsum(h); n=max(l.sum(),1); return float(h[0]),float(np.sum(c/(np.arange(len(h))+1)*h)/n),float(np.max(s[l>0])-np.max(s[l<=0])) if np.any(l<=0) else 0
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json')); val=[r for r in man['records'] if r['split']=='val' and r['kind']=='multi_positive_cross_video']; keys=sorted({r['query_track_key'] for r in val if r['query_track_key'] in meta}); smap={r['query_track_key']:[k for k in r.get('support_track_keys',[]) if k in meta] for r in val}; model=SafetyVectorBridge(); ck=torch.load(OUT/f'checkpoints/{TAG}_f{fold}_best.pt',map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); model.eval(); per={}; vids=np.array([meta[k]['video'] for k in keys]); cats=np.array([meta[k]['category'] for k in keys])
  for p in PREFIXES:
   mat=np.asarray([vec(k,meta,feats,p) for k in keys]); raw1=[];rawm=[];rawg=[];br1=[];brm=[];brg=[]; alpha=[]
   for i,k in enumerate(keys):
    cand=np.where((vids!=vids[i])&(np.arange(len(keys))!=i))[0]; lab=(cats[cand]==cats[i]).astype(float)
    if not lab.any(): continue
    q=mat[i]; sk=smap.get(k,[]); sv=np.asarray([mat[keys.index(s)] for s in sk if s in keys]); ctx=sv.mean(0) if len(sv) else q; ctx/=max(float(np.linalg.norm(ctx)),1e-8); raw=mat[cand]@q; support_quality=float((sv@q).max()) if len(sv) else 0
    with torch.no_grad(): z,a,_=model(torch.tensor(q).view(1,-1),torch.tensor(ctx).view(1,-1),torch.tensor([float(raw.max())]),torch.tensor([support_quality]),bool(len(sv))); z=z.numpy()[0]
    bridge=mat[cand]@z; x,y,g=calc(raw,lab); raw1.append(x);rawm.append(y);rawg.append(g); x,y,g=calc(bridge,lab); br1.append(x);brm.append(y);brg.append(g); alpha.append(float(a.item()))
   per[str(p)]={'raw':{'r1':float(np.mean(raw1)) if raw1 else 0,'map':float(np.mean(rawm)) if rawm else 0,'hard_gap':float(np.mean(rawg)) if rawg else 0},'bridge':{'r1':float(np.mean(br1)) if br1 else 0,'map':float(np.mean(brm)) if brm else 0,'hard_gap':float(np.mean(brg)) if brg else 0},'alpha_mean':float(np.mean(alpha)) if alpha else 0}
  folds.append({'fold':fold,'validation_tracklets':len(keys),'prefix':per})
 agg={str(p):{n:{m:float(np.mean([f['prefix'][str(p)][n][m] for f in folds])) for m in ('r1','map','hard_gap')} for n in ('raw','bridge')} for p in PREFIXES}; atomic(OUT/'metrics/gate_retrieval.json',{'protocol':'phase41_trained_safety_vector_bridge','folds':folds,'aggregate':agg,'row_vector_dim':768,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(OUT/'completion/retrieval.done',{'metrics':str(OUT/'metrics/gate_retrieval.json'),'p16':agg['16']}); print(json.dumps({'p16':agg['16']},indent=2))
if __name__=='__main__': main()
