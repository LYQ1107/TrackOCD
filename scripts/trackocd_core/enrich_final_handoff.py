#!/usr/bin/env python3
"""Export explicit per-seed four-order statistics and finalize report wording.

Posthoc public-result reporting only. Existing immutable scientific results
are never edited. Guard all prior handoff bindings before bounded doc edits.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.trackocd_core.report_final_handoff import mean_std, csv_text, p16_primary, validate
from src.trackocd_v2.io import atomic_json, atomic_write_text, sha256_file
OUT=ROOT/'outputs/trackocd_core';DOCS=ROOT/'docs/trackocd_core'


def per_seed_four_orders(val):
    groups=defaultdict(list)
    for c in val['cases']:groups[c['method'],c['prefix'],c['coverage_target'],c['seed']].append(c)
    rows=[]
    fields={'old_acc':'standard','new_acc':'standard','h_score':'standard','all_acc':'standard',
            'correct_commit_ct':'persistent','false_merge_rate':'persistent','false_split_new_rate':'persistent',
            'wait_unresolved_rate':'persistent','effective_commit_coverage':'persistent'}
    for (method,prefix,target,seed),cases in sorted(groups.items(),key=lambda x:str(x[0])):
        if len(cases)!=4 or {c['order'] for c in cases}!=set(val['fixed_full_GT']['reuse_opportunities']):
            raise ValueError('Every seed/point/prefix must retain all four actual orders')
        row=dict(scope=val['scope'],method=method,prefix=prefix,coverage_target=target,seed=seed,
                 video_orders=4,known_denominator=4413,novel_denominator=819,
                 CT_denominators_main_seed1027_seed1028_seed1029='527/529/547/531',
                 population_order_SD_not_confidence_interval=True,aliases_not_independent=True)
        for key,section in fields.items():
            row[key+'_four_order_mean'],row[key+'_four_order_std_ddof0']=mean_std([c[section][key] for c in cases])
        row['predicted_ID_commitment_coverage_four_order_mean'],row['predicted_ID_commitment_coverage_four_order_std_ddof0']=mean_std([c['runtime']['commitment_coverage'] for c in cases])
        rows.append(row)
    return rows


def main(commit):
    source='scripts/trackocd_core/enrich_final_handoff.py'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=commit or subprocess.check_output(['git','show',f'{commit}:{source}'])!=Path(__file__).read_bytes():raise ValueError('Register and verify exact report source first')
    result_path=OUT/'FINAL_RESULT.json';prior_sha=sha256_file(result_path);d=json.loads(result_path.read_text())
    if 'canonical_handoff_enrichment' in d or (OUT/'PER_SEED_FOUR_ORDER_RESULTS.csv').exists():raise ValueError('Never overwrite a completed enrichment')
    if d['status']!='COMPLETE_AMENDED_RESEARCH_NEGATIVE_NOT_STRONG_END_TO_END_SUCCESS':raise ValueError('Actual authorized negative handoff required')
    for group in ('artifact_sha256','prior_evidence_sha256'):
        for path,sha in d[group].items():
            if sha256_file(ROOT/path)!=sha:raise ValueError('Prior handoff changed: '+path)
    def read(n):return json.loads((OUT/(n+'.json')).read_text())
    core=read('CORE_TRAINING_FINAL_RESULT');rep=read('TRAIN_FIRST_REPRESENTATION_HELDOUT_RESULT');policy=read('TRAIN_FIRST_POLICY_HELDOUT_RESULT');val=read('LIMITED_MASA_EVALUATION_RESULT');validate(core,rep,policy,val)
    rows=per_seed_four_orders(val)
    if len(rows)!=510:raise ValueError('All 2040 cases/4 orders must be represented')
    old=(DOCS/'FINAL_REPORT.md').read_text()
    marker='''independent runs; actual mean H is **not H(mean Old, mean New)**. Each seed's
four-order mean and both seed / order+seed SD are in BASELINE_COMPARISON.csv;
each order/seed/prefix and full scope are in PER_VIDEO_ORDER_RESULTS.csv.'''
    replacement='''independent runs; actual mean H is **not H(mean Old, mean New)**.
PER_SEED_FOUR_ORDER_RESULTS.csv gives each seed's actual four-order mean and
four-order SD. BASELINE_COMPARISON.csv gives the mean of those seed means,
between-seed SD and descriptive combined order+seed SD. Individual order /
seed / prefix / full scope values remain in PER_VIDEO_ORDER_RESULTS.csv.'''
    point='All cases keep GT Known=4413, Novel=819.'
    if old.count(marker)!=1 or old.count(point)!=1:raise ValueError('Refuse unexpected report edits')
    risk=['### Main-point error and coverage audit (p16)', '',
          '| Method | False Merge % | unresolved reuse WAIT/missing % | actual reuse coverage % | all predicted-ID commitment % |',
          '|---|---:|---:|---:|---:|']
    for r in p16_primary(val['aggregate']):
        risk.append('| '+r['method']+' | '+' | '.join(f"{100*r[k+'_mean']:.4f}" for k in ('false_merge_rate','wait_unresolved_rate','effective_commit_coverage','predicted_ID_commitment_coverage'))+' |')
    risk+=['', 'Unresolved reuse includes missed physical opportunities, not only an explicit',
           'model WAIT action. All-ID commitment is separately reported; low mergerate',
           'with withheld/no-memory reuse is not a matched-coverage safety advantage.', '']
    updated=old.replace(marker,replacement).replace(point,'\n'.join(risk)+'\n'+point)
    train=(DOCS/'TRAINING_AND_REPRODUCTION.md').read_text()
    marker2='evaluate_limited_masa.py; report_core_training.py; report_final_handoff.py.'
    if train.count(marker2)!=1:raise ValueError('Preserve reproduction edits')
    train=train.replace(marker2,marker2+'\nenrich_final_handoff.py adds the explicit per-seed four-order reporting CSV,\nnot new evaluations; use the recorded separate source registration.')
    atomic_write_text(OUT/'PER_SEED_FOUR_ORDER_RESULTS.csv',csv_text(rows))
    atomic_write_text(DOCS/'FINAL_REPORT.md',updated)
    atomic_write_text(DOCS/'TRAINING_AND_REPRODUCTION.md',train)
    changed=['outputs/trackocd_core/PER_SEED_FOUR_ORDER_RESULTS.csv','docs/trackocd_core/FINAL_REPORT.md','docs/trackocd_core/TRAINING_AND_REPRODUCTION.md']
    for path in changed:d['artifact_sha256'][path]=sha256_file(ROOT/path)
    d['prior_evidence_sha256'][source]=sha256_file(Path(__file__).resolve())
    d['canonical_handoff_enrichment']={'source_registration_commit':commit,'source_exact_remote_verified_before_execution':True,
        'source_sha256':sha256_file(Path(__file__).resolve()),'prior_handoff_result_sha256':prior_sha,
        'actual_per_seed_four_order_rows':510,'new_experiments':False,
        'purpose':'Explicit per-seed four-order mean/SD and primary FM/WAIT/coverage disclosure; no scientific result or frozen source changed'}
    atomic_json(result_path,d)
    print(json.dumps(d['canonical_handoff_enrichment']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preregistration-commit',required=True);main(p.parse_args().preregistration_commit)
