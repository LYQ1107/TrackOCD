#!/usr/bin/env python3
"""TRAIN-only candidate rank audit for the registered Phase85 support manifest."""
from __future__ import annotations
import datetime as dt, json, os, tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase85'
MAN=OUT/'manifests/phase85_support_prefix_manifest.json'; DATA=OUT/'manifests/phase85_support_prefix_features.npz'
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=str(path.parent))
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def main():
 m=json.loads(MAN.read_text()); d=np.load(DATA,allow_pickle=False); targets=d['targets']; off=d['offsets']; meta=m['groups_meta']
 ks=(4,8,16,32); rows=[]
 for fold_name,fd in m['folds'].items():
  for split,ids in [('fit',fd['fit_groups']),('validation',fd['validation_groups'])]:
   for p in (1,2,4,8,16):
    sub=[i for i in ids if int(meta[i]['prefix'])==p]; pos=[i for i in sub if int(targets[i])<32]; rec={}
    for k in ks:
     hit=sum(int(targets[i])<k for i in pos); rec[str(k)]={'hit':hit,'denominator':len(pos),'recall':hit/len(pos) if pos else None}
    rows.append({'fold':int(fold_name),'split':split,'prefix':p,'groups':len(sub),'positive_groups':len(pos),'defer_groups':len(sub)-len(pos),'recall_at_k':rec})
 # aggregate across source buckets; report exact numerators and no event labels
 agg={}
 for split in ('fit','validation'):
  for p in (1,2,4,8,16):
   sub=[r for r in rows if r['split']==split and r['prefix']==p]; pos=sum(r['positive_groups'] for r in sub); z={'groups':sum(r['groups'] for r in sub),'positive_groups':pos,'defer_groups':sum(r['defer_groups'] for r in sub),'recall_at_k':{}}
   for k in ks:
    hit=sum(r['recall_at_k'][str(k)]['hit'] for r in sub); z['recall_at_k'][str(k)]={'hit':hit,'denominator':pos,'recall':hit/pos if pos else None}
   agg[f'{split}_p{p}']=z
 valid16=agg['validation_p16']['recall_at_k']['16']['recall'] or 0.; chosen=16 if valid16>=.90 else 32
 out={'schema_version':'trackocd.phase85.support_topk_audit.v1','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'manifest':str(MAN.resolve()),'data':str(DATA.resolve()),'rows':rows,'aggregate':agg,'registered_default_k':chosen,'selection_rule':'validation_p16_recall>=0.90 -> K=16 else K=32','public_dev_q1_sealed_accessed':False}
 atomic(OUT/'audit/support_topk_audit.json',out); atomic(OUT/'completion/support_topk_audit.done',{'status':'completed','created_utc':out['created_utc'],'default_k':chosen}); print(json.dumps({'default_k':chosen,'validation_p16':agg['validation_p16'],'validation_rows':len(rows)},indent=2,sort_keys=True))
if __name__=='__main__':main()
