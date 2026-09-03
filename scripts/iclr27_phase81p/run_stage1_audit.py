#!/usr/bin/env python3
"""Materialize the Phase81P read-only physical audit summary."""
from __future__ import annotations
import datetime, hashlib, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(p): return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
def main():
 call=load(ROOT/'outputs/iclr27_phase81p/audit/q0_physical_callgraph.json'); tax=load(ROOT/'outputs/iclr27_phase81p/audit/assignment_failure_taxonomy.json'); obs=load(ROOT/'outputs/iclr27_phase80c/audit/observability_quality_audit.json'); byp={str(x['prefix']):{'source_reliable':sum(1 for r in x['records'] if r.get('source_reliable')),'target_reliable':sum(1 for r in x['records'] if r.get('target_reliable')),'both_reliable':sum(1 for r in x['records'] if r.get('both_reliable'))} for x in []}
 records=obs['records'];
 for pref in obs['prefixes']:
  rr=[r for r in records if r.get('prefix')==pref and r.get('polarity')=='positive']; byp[str(pref)]={'events':len(rr),'source_reliable':sum(bool(r['source_reliable']) for r in rr),'target_reliable':sum(bool(r['target_reliable']) for r in rr),'both_reliable':sum(bool(r['both_reliable']) for r in rr)}
 result={'schema_version':'phase81p.stage1_audit.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'q0_callgraph_sha256':hashlib.sha256((ROOT/'outputs/iclr27_phase81p/audit/q0_physical_callgraph.json').read_bytes()).hexdigest(),'taxonomy_sha256':hashlib.sha256((ROOT/'outputs/iclr27_phase81p/audit/assignment_failure_taxonomy.json').read_bytes()).hexdigest(),'q0_baseline_prefixes':byp,'p16_reference':{'source_pool_good':72,'target_pool_good':64,'source_reliable':49,'target_reliable':40,'both_reliable':25,'assignment_gap_events':36},'assignment_taxonomy':tax['taxonomy_counts'],'training_started':False,'held_dev_q1_public_sealed_accessed':False,'next_action':'run TRAIN manifest smoke then association smoke'}
 atomic(ROOT/'outputs/iclr27_phase81p/audit/stage1_summary.json',result); atomic(ROOT/'outputs/iclr27_phase81p/audit/stage1.done',{'status':'PASS_READ_ONLY_Q0_AUDIT','created_utc':result['created_utc']}); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
