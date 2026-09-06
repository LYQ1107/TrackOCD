#!/usr/bin/env python3
"""Materialize the TRAIN-only OOF meta rows used by U1 (no diagnostic inputs)."""
from __future__ import annotations
import hashlib, json, os, tempfile, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.iclr27_phase86.run_u1_selective import build_table, MAN, OUT

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 m, rows, experts=build_table(); p=OUT/'manifests/u1_oof_meta.jsonl'; p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f:
   for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
   f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
 q=OUT/'manifests/u1_oof_meta_manifest.json'; obj={'schema_version':'trackocd.phase86.u1_oof_meta_manifest.v1','rows':len(rows),'feature_dim':64,'meta_path':str(p.resolve()),'meta_sha256':sha(p),'source_phase85_manifest':str(MAN.resolve()),'source_phase85_manifest_sha256':sha(MAN),'diagnostic_paths_consumed':[],'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_or_text_as_model_input':False,'labels_posthoc_only':True,'utility_labels':{'help':1,'harm':-4,'both_correct_or_wrong':0}}
 with q.open('w') as f: json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(obj,indent=2,sort_keys=True))
if __name__=='__main__':main()
