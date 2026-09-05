#!/usr/bin/env python3
"""Low-frequency activity watchdog; never sleeps or finalizes early."""
from __future__ import annotations
import datetime as dt, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs/iclr27_phase85"
def parse(s): return dt.datetime.fromisoformat(s.replace("Z","+00:00"))
def main():
 reg=json.loads((OUT/"audit/window_registration.json").read_text()); now=dt.datetime.now(dt.timezone.utc); deadline=parse(reg["deadline_utc"]); remaining=(deadline-now).total_seconds(); files=[]
 for d in (OUT/"audit",OUT/"metrics",OUT/"completion"):
  if d.exists(): files.extend(p for p in d.rglob("*") if p.is_file())
 latest=max((p.stat().st_mtime for p in files),default=0); age=now.timestamp()-latest if latest else None
 workers=[]
 for p in OUT.rglob("*.launched") if OUT.exists() else []:
  done=p.with_suffix(".done")
  if not done.exists(): workers.append(str(p))
 status="ACTIVE" if remaining<=3600 or age is None or age<=2700 else "RESEARCH_IDLE"
 out={"schema_version":"trackocd.phase85.activity.v1","now_utc":now.isoformat(),"remaining_seconds":remaining,"status":status,"latest_artifact_age_seconds":age,"unfinished_markers":workers,"next_action":"continue highest-information registered repair/support stage" if status=="RESEARCH_IDLE" else "continue registered work or finalize only after unlock","finalization_allowed":remaining<=2700,"public_dev_q1_sealed_accessed":False}
 print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":main()
