#!/usr/bin/env python3
"""Complete the metric summary from frozen D0/D1 traces without rerunning them."""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase86'
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t): os.unlink(t)
def main():
 d=json.loads((OUT/'diagnostic_ocd/summary.json').read_text()); streams={}
 for name,stream in d['streams'].items():
  if name.startswith('D2') or name.startswith('D3'): continue
  folds=[]
  for f in stream['folds']:
   m=f['metrics']; folds.append({'fold':f['fold'],'events':f['events'],'commit_ct':m['commit_ct'],'existing_f1':m['existing_f1'],'existing_f1_macro':m['existing_f1_macro'],'existing_precision':m['existing_precision'],'existing_recall':m['existing_recall'],'negative_false_merge_rate':m['negative_false_merge_rate'],'false_merge_rate_macro':m['false_merge_rate_macro'],'unresolved_rate':m['unresolved_rate'],'premature_rate':m['premature_rate'],'pre_prefix_defer_rate':m['pre_prefix_defer_rate'],'fragmentation_rate_macro':m['fragmentation_rate_macro'],'duplicate_births':m['duplicate_births'],'selection_score':m['selection_score'],'category_coverage':m['category_coverage'],'video_coverage':m['video_coverage'],'known_micro':m['known_micro'],'known_macro':m['known_macro'],'new_precision':m['new_precision'],'new_recall':m['new_recall'],'new_f1_macro':m['new_f1_macro'],'novel_nmi_macro':m['novel_nmi_macro']})
  numeric=['existing_f1','existing_f1_macro','existing_precision','existing_recall','negative_false_merge_rate','false_merge_rate_macro','unresolved_rate','premature_rate','pre_prefix_defer_rate','fragmentation_rate_macro','known_micro','known_macro','new_precision','new_recall','new_f1_macro','novel_nmi_macro','selection_score']
  agg={k:sum(float(x[k]) for x in folds)/max(1,len(folds)) for k in numeric}; agg['duplicate_births']=sum(int(x['duplicate_births']) for x in folds); agg['category_coverage']=sum(int(x['category_coverage']) for x in folds); agg['video_coverage']=sum(int(x['video_coverage']) for x in folds)
  streams[name]={'aggregate_commit_ct':stream['aggregate_commit_ct'],'folds':folds,'aggregate':agg,'denominator':d['denominator'],'diagnostic_only':True,'controller_frozen':stream['controller_frozen']}
 out={'schema_version':'trackocd.phase86.diagnostic_ocd_full_metrics.v1','phase':86,'label':'DIAGNOSTIC_ONLY_DO_NOT_SELECT','source_summary':str((OUT/'diagnostic_ocd/summary.json').resolve()),'streams':streams,'public_dev_q1_sealed_accessed':False,'sealed_run':False,'formal_ocd_run':False}
 atom(OUT/'audit/diagnostic_ocd_full_metrics.json',out); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
