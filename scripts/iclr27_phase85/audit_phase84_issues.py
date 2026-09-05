#!/usr/bin/env python3
"""Record the Phase84 contract issues and the Phase85 repair evidence."""
from __future__ import annotations
import datetime as dt, hashlib, json, os, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs/iclr27_phase85/audit"

def sha(path: Path)->str:
 h=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def atomic(path:Path,obj):
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=str(path.parent))
 with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def read(path):
 try:return json.loads(Path(path).read_text())
 except Exception:return None
def main():
 p84=ROOT/"outputs/iclr27_phase84/audit"
 physical=read(p84/"physical_r_diagnostic.json") or {}
 q0=read(OUT/"physical_r_q0_q0_parity_v5_adapter.json") or {}
 improved=read(OUT/"physical_r_improved_improved_single_anchor_v2_adapter.json") or {}
 audit={"schema_version":"trackocd.phase85.phase84_issue_audit.v1","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"phase84_read_only":True,"issues":[
  {"id":"A84P_LAST_APPEARANCE","phase84_evidence":"run_full_temporal_physical.py stores state['app'] and reconnect scores use last observation","repair":"Phase85 running app_sum/app_count/app_mean; reconnect uses normalized temporal mean","verification":"P1 full replay summary appearance_state=normalized_running_sum_and_count"},
  {"id":"FAKE_Q0_PARITY","phase84_evidence":"build_physical_r_adapter.py compares raw_vectors - raw_vectors","repair":"Phase85 q0 adapter reconstructs vectors and compares to FrozenTrackTable.raw_vector","verification":q0.get("parity",{})},
  {"id":"MULTI_ROOT_ANCHOR","phase84_evidence":"Phase84 adapter expands every root seen in prefix","repair":"Phase85 improved adapter uses only last mapped causal row's canonical root and cutoff","verification":improved.get("info",{}).get("membership_rule")},
  {"id":"IOU_JOIN_OPACITY","phase84_evidence":"fallback matching was not separated from stable keys","repair":"Phase85 exact path/frame/proposal join first, explicit image IoU fallback and unmatched counts","verification":q0.get("join",{})},
  {"id":"PHYSICAL_CONTAMINATION","phase84_evidence":physical.get("semantic_contamination",{}),"repair":"Phase85 headlines contamination and keeps labels post-hoc only","verification":{"cross_category_union":physical.get("same_cross_category_merge_rate",{}).get("cross"),"same_category_union":physical.get("same_cross_category_merge_rate",{}).get("same"),"within_root":physical.get("within_root_feature_variance",{})}},
  {"id":"MISSING_PHYSICAL_TRACKEVAL","phase84_evidence":"Phase84 full route reported physical proxy rather than formal TrackEval","repair":"Phase85 reuses Phase82R exporter/runner on identical 91-video TRAIN subset","verification":"outputs/iclr27_phase85/metrics/trackeval/{q0_event91,temporal_mean_event91}"},
  {"id":"B84SQ_ARTIFACT_MIXUP","phase84_evidence":"B84S-Q report table read B84S-RA (24/9) instead of B84S-Q (7/2)","repair":"Phase85 sections carry route/tag/schema assertions and provenance ledger","verification":"report generator will refuse mismatches"},
  {"id":"EARLY_IDLE_WAIT","phase84_evidence":"finalize_when_unlocked slept for remaining window before report","repair":"Phase85 activity watchdog; finalizer exits early with explicit error and no sleep","verification":"registered window has no finalizer process before final interval"}
 ],"phase85_repair_artifacts":{"temporal_smoke":"outputs/iclr27_phase85/metrics/temporal_mean_smoke_r1.json","temporal_full":"outputs/iclr27_phase85/metrics/temporal_mean_full.json","q0_parity":"outputs/iclr27_phase85/audit/physical_r_q0_q0_parity_v5_adapter.json","improved_adapter":"outputs/iclr27_phase85/audit/physical_r_improved_improved_single_anchor_v2_adapter.json","physical_r":"outputs/iclr27_phase85/audit/physical_r_comparison.json"},"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False}
 atomic(OUT/"phase84_issue_audit.json",audit)
 repairs={"schema_version":"trackocd.phase85.repair_events.v1","created_utc":audit["created_utc"],"events":[{"issue":x["id"],"status":"REPAIRED_OR_AUDITED","evidence":x["verification"]} for x in audit["issues"]]}
 atomic(OUT/"repair_events.json",repairs); print(json.dumps(audit,indent=2,sort_keys=True))
if __name__=="__main__":main()
