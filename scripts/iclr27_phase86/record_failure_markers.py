#!/usr/bin/env python3
from __future__ import annotations
import datetime as dt,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase86/completion'
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def main():
 now=dt.datetime.now(dt.timezone.utc).isoformat(); atom(OUT/'u1_final_alltrain_v1.failed',{'status':'FAILED','route':'U1_FINAL_UTILITY_GATE','reason':'first all-TRAIN attempt wrote checkpoint but failed in sha helper before metrics/done; superseded by final_alltrain_v1_fix1','time_utc':now}); print('recorded u1 final failed marker')
if __name__=='__main__':main()
