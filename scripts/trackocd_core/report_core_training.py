#!/usr/bin/env python3
"""Final synthesis of actually completed frozen experiments, never new fits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,atomic_write_text,sha256_file
RUNNING_PROTOCOL_STATUS=('The common predicted-track features are complete. Frozen semantic inference\n'
    'and sealed-case posthoc metrics are in progress; the full matrix is not yet\n'
    'complete. This document preserves the protocol registered before execution,\n'
    'not invented final Val results.')


def require_complete(rep,policy,val):
    if len(rep['cases'])!=420 or len(policy['cases'])!=1800:raise ValueError('Real controlled matrix incomplete')
    if val['status']!='COMPLETE_REAL_FULL_LIMITED_MASA_FROZEN_SEMANTIC_EVALUATION':raise ValueError('Full predicted metrics incomplete')
    if (val['unique_sealed_executions'],val['logical_cases_including_explicit_aliases'],len(val['cases']),len(val['paired_comparisons']))!=(840,2040,2040,780):raise ValueError('Do not substitute partial or aliased independent experiments')
    if any(r['standard']['old_denominator']!=4413 or r['standard']['new_denominator']!=819 for r in val['cases']):raise ValueError('Full GT penalties mandatory')


def percent(v):return f'{100*float(v):.2f}'


def table(rows,name):
    text=['| '+name+' | Old % | New % | mean H % | pure CT % | False Merge % | reuse coverage % |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for label,r in rows:
        text.append('| '+label+' | '+' | '.join(percent(r[k+'_mean']) for k in ('old_acc','new_acc','h_score','correct_commit_ct','false_merge_rate','effective_commit_coverage'))+' |')
    return '\n'.join(text)+'\n'


def require_terminal(receipt,returncodes_key,adopted_groups=()):
    codes=receipt.get(returncodes_key)
    if receipt.get('error') is not None or not isinstance(codes,list) or any(type(c) is not int or c!=0 for c in codes):
        raise ValueError('Actual successful owned-child terminal receipt required')
    if receipt.get('legacy_live_workers_left_untouched'):
        raise ValueError('Cannot finalize while adopted workers are still live')
    proofs=receipt.get('adopted_original_workers_terminal_proof',[])
    if len(proofs)!=len(adopted_groups) or {r['group'] for r in proofs}!=set(adopted_groups):
        raise ValueError('Every adopted worker needs terminal proof; no fabricated zero exit')
    if any(r.get('orphan_exit_status')!='UNAVAILABLE_NOT_A_CHILD' or r.get('atomic_terminal_and_every_seal_verified') is not True for r in proofs):
        raise ValueError('Orphan exit status unknown; atomic completion proof mandatory')


def terminal_provenance(matrix,inference,val):
    """Read only completed posthoc/operational receipts, never inference inputs."""
    plan=json.loads((matrix/'plan.json').read_text());recovery_name=inference.get('scheduling_recovery_receipt')
    original=[inference['preregistration_commit']];recovery_commit=None;hashes={};adopted=[]
    if recovery_name:
        match=re.fullmatch(r'recovery_([0-9a-f]{32})_started\.json',recovery_name)
        if not match:raise ValueError('Bounded recovery receipt filename required')
        receipt_path=matrix/recovery_name
        if receipt_path.is_symlink():raise ValueError('No recovery receipt symlink')
        recovery=json.loads(receipt_path.read_text());attempt=match.group(1)
        if recovery['identity']!=inference['identity'] or recovery['plan_sha256']!=inference['plan_sha256'] or recovery['preregistration_commit']!=inference['preregistration_commit']:
            raise ValueError('Recovery changed original frozen inference identity')
        recovery_commit=recovery['preregistration_commit']
        if any(a['identity']!=inference['identity'] or a['plan_sha256']!=inference['plan_sha256'] for a in recovery['prior_assignments'].values()):
            raise ValueError('Original adopted assignments must retain the same prospective inputs')
        for name,sha in recovery['source_sha256'].items():
            if hashlib.sha256(subprocess.check_output(['git','show',f'{recovery_commit}:{name}'])).hexdigest()!=sha:
                raise ValueError('Actual recovery source not registered at recorded commit')
        original=sorted({a['preregistration_commit'] for a in recovery['prior_assignments'].values()}) or [recovery_commit]
        adopted=[w['group'] for w in recovery['adopted_current_host_workers']]
        hashes[recovery_name]=sha256_file(receipt_path)
    else:
        attempt=json.loads((matrix/'inference_supervisor_progress.json').read_text())['attempt']
        if not re.fullmatch('[0-9a-f]{32}',attempt):raise ValueError('Actual parent attempt required')
    path=matrix/f'supervisor_{attempt}.json';terminal=json.loads(path.read_text())
    require_terminal(terminal,'owned_child_returncodes',adopted)
    if len(terminal['completed_groups'])!=len(plan['groups']) or set(terminal['completed_groups'])!=set(plan['groups']):
        raise ValueError('All registered groups need actual terminal completion')
    hashes[path.name]=sha256_file(path)
    evaluator_receipts=[]
    for path in matrix.glob('evaluator_supervisor_*.json'):
        match=re.fullmatch('evaluator_supervisor_([0-9a-f]{32})\\.json',path.name)
        if not match:continue
        assignment_path=matrix/f'evaluator_assignment_{match.group(1)}.json'
        assignment=json.loads(assignment_path.read_text())
        if assignment['preregistration_commit']!=val['preregistration_commit'] or assignment['source_sha256']!=val['source_sha256']:continue
        require_terminal(json.loads(path.read_text()),'owned_returncodes')
        evaluator_receipts.append(path.name);hashes[path.name]=sha256_file(path);hashes[assignment_path.name]=sha256_file(assignment_path)
    if not evaluator_receipts:raise ValueError('Actual full evaluator terminal receipt required')
    case_registration=set();records={r['job']['id']:r for r in inference['records']}
    for key in plan['jobs']:
        metric=json.loads((matrix/'cases'/key/'case_metrics.json').read_text())
        if metric['status']!='COMPLETE_FULL_GT_POSTHOC_SEMANTIC_METRICS' or metric['input_identity']!=inference['identity'] or metric['evaluator_source_sha256']!=val['source_sha256'] or metric['sealed_ledger_sha256']!=records[key]['ledger']['sha256']:
            raise ValueError('Every actual posthoc case must bind the unchanged all-ID seal/source')
        case_registration.add(metric['evaluator_preregistration_commit'])
    return {'original_frozen_inference_registration_commits':original,'scheduling_recovery_registration_commit':recovery_commit,
            'evaluator_collector_registration_commit':val['preregistration_commit'],'actual_case_evaluator_registration_commits':sorted(case_registration),
            'all_original_and_recovered_scientific_inputs_identical':True,'adopted_worker_exit_status_not_fabricated':True,
            'actual_successful_terminal_receipts_sha256':hashes}


def main(commit):
    out=ROOT/'outputs/trackocd_core';base=out/'core_training/train_first_v1';docs=ROOT/'docs/trackocd_core'
    def read(name):return json.loads((out/(name+'.json')).read_text())
    rep=read('TRAIN_FIRST_REPRESENTATION_HELDOUT_RESULT');policy=read('TRAIN_FIRST_POLICY_HELDOUT_RESULT');val=read('LIMITED_MASA_EVALUATION_RESULT');require_complete(rep,policy,val)
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=commit or subprocess.check_output(['git','show',f'{commit}:scripts/trackocd_core/report_core_training.py'])!=Path(__file__).read_bytes():raise ValueError('Final report source must be registered and exact remote')
    old=(docs/'LIMITED_MASA_EVALUATION.md').read_text()
    if '## Completed actual full semantic evaluation' in old or old.count(RUNNING_PROTOCOL_STATUS)!=1:
        raise ValueError('Preserve existing report edits/completion; exact pending status required')
    freeze=json.loads((base/'model_freeze.json').read_text())
    for path,sha in freeze['protected_sha256'].items():
        if sha256_file(ROOT/path)!=sha:raise ValueError('Frozen Train model/protocol changed')
    inference=json.loads((base/'limited_masa_evaluation/full_inference_manifest.json').read_text())
    if sha256_file(base/'limited_masa_evaluation/full_inference_manifest.json')!=val['full_inference_manifest_sha256'] or inference['unique_executions']!=840:raise ValueError('Actual inference lineage lost')
    lineage=terminal_provenance(base/'limited_masa_evaluation',inference,val)
    checkpoints=[];retained=[];training={}
    for family in ('representation','evidence','policy'):
        path=base/family/'training_receipt.json';receipt=json.loads(path.read_text());training[family]={'receipt_path':str(path),'receipt_sha256':sha256_file(path),'preregistration_commit':receipt['preregistration_commit'],
            'actual_fits':len(receipt['fits']),'resources':receipt['resources'],'checkpoint_bytes':receipt.get('checkpoint_bytes',receipt['resources'].get('checkpoint_bytes')),
            'full_fit_sample_coverage_not_selected_early_checkpoint_footprint':[
                {'model':f['model'],'seed':f['seed'],'fit_tracks_available':f.get('fit_tracks_available'),
                 'distinct_fit_tracks_actually_seen_over_full_budget':f.get('fit_tracks_actually_seen')} for f in receipt['fits']]}
        for fit in receipt['fits']:
            for cp in fit['checkpoints']:
                r=cp['checkpoint'];p=ROOT/r['path']
                if sha256_file(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('Actual local checkpoint incomplete')
                retained.append({'path':str(p),'sha256':r['sha256'],'bytes':r['bytes']})
            selected=fit['selected_checkpoint'];cp=selected['checkpoint']
            checkpoints.append({'phase':family,'model':fit['model'],'seed':fit['seed'],'representation':fit.get('representation'),
                'step_or_epoch':selected.get('step',selected.get('epoch')),'local_path':str(ROOT/cp['path']),'sha256':cp['sha256'],'bytes':cp['bytes']})
    assert len(checkpoints)==33 and len(retained)==99
    train_rows=[]
    for backend in ('B0_frame_snapshot_vote','B1_track_nearest','B2_track_dpmeans'):
        train_rows.append((backend,next(r for r in rep['aggregate'] if r['representation']=='A0_RAW' and r['backend']==backend and r['prefix']==16)))
    for name in ('A1_SELECTED','A2_EVIDENCE','A2_CAPACITY_CONTROL'):
        train_rows.append((name,next(r for r in rep['aggregate'] if r['representation']==name and r['backend']=='B1_track_nearest' and r['prefix']==16)))
    for name in ('D1_SIMPLE_MLP','D2_RISK_AWARE','WITHOUT_RISK','FULL','WITHOUT_WAIT','WITHOUT_MEMORY','RESET_PER_VIDEO'):
        train_rows.append((name,next(r for r in policy['aggregate'] if r['method']==name and r['prefix']==16 and r['coverage_target']==1.)))
    val_rows=[(r['method'],r) for r in val['aggregate'] if r['prefix']==16 and r['coverage_target'] in (None,1.)]
    primary_pairs=[r for r in val['paired_comparisons'] if r['prefix']==16 and r['coverage_target']==1.]
    stable=[]
    for left,right in sorted({(r['left'],r['right']) for r in primary_pairs}):
        pairs=[r for r in primary_pairs if (r['left'],r['right'])==(left,right)]
        stable.append({'left':left,'right':right,'cases':len(pairs),'H_positive':sum(r['h_score_delta']>0 for r in pairs),
            'CT_positive':sum(r['CT_delta']>0 for r in pairs),'matched_coverage_cases':sum(r['comparable_within_tolerance'] for r in pairs),
            'matched_CT_gain_or_FM_reduction_without_more_wrongKnown':sum(r['comparable_within_tolerance'] and (r['CT_delta']>0 or r['false_merge_delta']<0) and r['wrong_known_all_novel_delta']<=0 for r in pairs)})
    corrections=read('TRAIN_FIRST_ERROR_CORRECTION_RESULT');roles=read('TRAIN_FIRST_REPRESENTATION_ROLE_ANALYSIS');shift=read('INPUT_SHIFT_DIAGNOSTIC_RESULT')
    if shift['status']!='COMPLETE_POSTHOC_DESCRIPTIVE_INPUT_SHIFT_NOT_CAUSAL_INTERVENTION' or shift['model_freeze_sha256']!=sha256_file(base/'model_freeze.json'):raise ValueError('Actual unchanged descriptive shift evidence required')
    support=read('audit/physical_cross_video_support')['frontends']['MASA_NATIVE']
    physical_rows=[]
    for order,r in support['orders'].items():
        physical_rows.append({'order':order,'fixed_GT_opportunities':r['fixed_gt_reuse_opportunities'],'current_reliable':r['reliable_current_opportunities'],
            'missing_current':r['missing_current_physical_opportunities'],'both_any_observed_prefix':r['prefix_support']['1']['both_earlier_and_current_have_at_least_p'],
            'both_at_least_16':r['prefix_support']['16']['both_earlier_and_current_have_at_least_p']})
    for name in ('PER_ORDER_RESULTS.csv','ERROR_BREAKDOWN.csv'):
        rows=list(csv.DictReader((out/name).open()))
        assert len(rows)==4260 and sum(r['stage']=='T5' for r in rows)==2040
    local_bytes=sum(p.stat().st_size for path in (base,out/'features/train_first_v1',out/'features/masa_limited_prefix_v1') for p in path.rglob('*') if p.is_file())
    if local_bytes>15*2**30:raise RuntimeError('Soft new-storage budget exceeded; never delete historical assets')
    resource={'actual_training':training,'full_features':read('LIMITED_MASA_FEATURE_FULL_RESULT'),
        'full_inference_wall_seconds':inference['wall_seconds'],'peak_inference_case_RSS_bytes':max(r['resources']['peak_RSS_bytes'] for r in inference['records']),
        'peak_inference_case_GPU_reserved_bytes':max(r['resources']['peak_GPU_reserved_bytes'] for r in inference['records']),
        'posthoc_evaluation':val['resource_summary'],'conservative_task_private_file_bytes':local_bytes,
        'storage_soft_bytes':15*2**30,'storage_hard_bytes':30*2**30,'system_RAM_reserve_fraction':.25,'foreign_process_interference':False}
    artifact_names=['TRAIN_FIRST_REPRESENTATION_RESULT.json','TRAIN_FIRST_EVIDENCE_RESULT.json','TRAIN_FIRST_POLICY_RESULT.json',
        'TRAIN_FIRST_REPRESENTATION_HELDOUT_RESULT.json','TRAIN_FIRST_POLICY_HELDOUT_RESULT.json','TRAIN_FIRST_REPRESENTATION_ROLE_ANALYSIS.json',
        'TRAIN_FIRST_ERROR_CORRECTION_RESULT.json','TRAIN_FIRST_MODEL_FREEZE_RESULT.json','LIMITED_MASA_FEATURE_FULL_RESULT.json','LIMITED_MASA_FULL_FEATURE_VALIDATION.json',
        'EXACT_SEARCH_CPU_PROOF.json','EXACT_SEARCH_CUDA_PROOF.json','TRAINING_CURVES_RESULT.json','LIMITED_MASA_EVALUATION_RESULT.json',
        'TRAIN_BASELINE_COMPARISON.csv','REPRESENTATION_ABLATION.csv','PERSISTENT_POLICY_COMPARISON.csv','PER_ORDER_RESULTS.csv','ERROR_BREAKDOWN.csv',
        'ERROR_CORRECTION_COMPARISON.csv','TRAINING_LOSS_TRACES.csv','LIMITED_MASA_CASE_ALIAS_MANIFEST.csv','INPUT_SHIFT_DIAGNOSTIC_RESULT.json']
    artifact_hashes={n:sha256_file(out/n) for n in artifact_names}
    result={'schema_version':'trackocd.core.actual-train-first-final.v1','status':'COMPLETE_REAL_TRAIN_AND_LIMITED_PREDICTED_EVALUATION_SCIENTIFIC_NEGATIVE',
        'scope':'Independent new Train-first main, not R2; limited annotated-cadence MASA replay',
        'all_T0_through_T6_scientific_work_complete':True,'historical_M1_status':'BLOCKED_FRONTEND_QUALITY',
        'scope_authorization_sha256':'cc874517b5366c019b625ce3d6fcf0a33333c359cfb1dd8d03039d2cde290e99',
        'report_source_registration_commit':commit,'source_registration_exact_remote_verified':True,'full_inference_preregistration_commit':inference['preregistration_commit'],
        'posthoc_evaluator_registration_commit':val['preregistration_commit'],
        'original_registration_and_operational_recovery_provenance':lineage,
        'report_source_sha256':sha256_file(Path(__file__).resolve()),'model_freeze_sha256':sha256_file(base/'model_freeze.json'),'protected_sha256':freeze['protected_sha256'],
        'actual_fits':33,'selected_checkpoints':checkpoints,'retained_checkpoints':retained,'controlled_representation_cases':420,'controlled_policy_cases':1800,
        'limited_MASA_unique_executions':840,'limited_MASA_logical_cases':2040,'canonical_per_order_rows':4260,'actual_loss_trace_rows':15360,
        'train_p16_aggregate':[{'method':name,**row} for name,row in train_rows],'limited_MASA_p16_aggregate':[row for _,row in val_rows],
        'limited_MASA_p16_fixed_comparison_sign_counts':stable,'Native_physical_support':physical_rows,'posthoc_input_shift':shift,'identical_frozen_physical_scores_for_all_methods':val['physical_native_reference'],
        'scientific_conclusions':{'Representation_PASS':False,'Temporal_Evidence_PASS':False,'Persistent_Decision_PASS':False,
            'reason':'Unseen Train macroRank1 not improved; A2 not better than staticcapacity; zero pureCT in all controlled learned policy cases. Report Val partial findings separately.',
            'new_TrackOCD_method_paper_claim_supported':False,'complete_credible_negative_result':True},
        'GT_in_live_memory':False,'Val_training_or_tuning':False,'TAO_Test_access':False,'new_physical_inference':False,'new_downloads':False,
        'PHE':val['PHE'],'artifact_sha256':artifact_hashes,'resources':resource,
        'old_app_goal_is_not_falsely_completed':True,'final_delivery_remote_verification':'Performed after report commit; see actual handoff, not a self-referential invented commit SHA'}
    checkpoint_table='| Phase / representation / model / seed | selected step or epoch | actual local checkpoint | SHA256 |\n|---|---:|---|---|\n'
    for cp in checkpoints:checkpoint_table+=f"| {cp['phase']} / {cp['representation'] or '-'} / {cp['model']} / {cp['seed']} | {cp['step_or_epoch']} | `{cp['local_path']}` | `{cp['sha256']}` |\n"
    questions=f'''# Scientific conclusion — real negative result

1. **Why previous Known collapse?** Old R0/R1 had tiny four-class/12-track
   fitting, poorer prototype-score AUROC and heavy wrong-Known assignment.
   This implicates learned open-set geometry as well as thresholds; it does
   not uniquely prove a loss/capacity cause. Old results are unchanged and
   numerically incomparable with this independent enlarged protocol.
2. **Did more legal Train improve open-set geometry?** Overall p16 Rank1 and
   Known/pseudo AUROC improve, but primarily on fitting Known. Four sparse
   heldout pseudo categories / 16 queries all have cross-video positives;
   pseudo macroRank1 raw .3500 vs A1/A2 .3083 at p16, and learned top1 is lower
   at all five prefixes. Partial Recall@K/AUROC gains do not make a stable
   unseen-geometry PASS. Effective rank falls roughly54.18→8.59.
3. **Is Adapter better than frozen DINO?** Known recognition and Standard H
   improve in controlled experiments, not universal unseen/persistent
   discovery. See actual tables, role analysis and paired errors. A1 fixes
   many Known errors but pseudo-Novel net correct count decreases1.167 per
   p16case. Do not turn Seen-Known probe improvement into Novel success.
4. **Independent temporal contribution?** Not supported: A2 p16 H .5136 vs
   same-capacity static .5413 and A1 .5276, CT .0794 vs static .0899 / A1
   .1693; A1→A2 adds26.083 errors while fixing5.583, whereas static introduces
   fewer errors. These are mean counts, not independent categories.
5. **Does risk-aware decision beat MLP?** Not established: every1800controlled
   learned-policy case has zero pureCT. D2 lower wrongKnown can accompany
   contaminated merging; matched coverage is mandatory. Full Val fixed
   pair sign counts below are separate observations, not a rescue fit.
6. **WAIT safer or merely rejection?** Train noWAIT FM .3333 vs FULL .3003
   while reuse coverage1→.9669 and all-track commitment1→.9152. This alone is
   not a matched-risk benefit. Full Val reports both actual coverage gaps;
   no favorable interpolation or infinite WAIT. NoMemory/reset remove reuse,
   so their low mergerate is not safe persistent discovery by itself.
7. **Cross-individual/video anonymous reuse?** Audited raw baselines can;
   learned policy has zero pureCT on controlled holdout. Offline Standard
   Hungarian can still count an impure token as New-correct, so that score
   is not live correct memory reuse. All840fullVal seals/2040logical reports
   are completed, including unknown/unmatched memory members.
8. **GT vs predicted gap?** Compare the completed tables below. Train207/16
   targets vs full Val4413/819, category support, physical noise/shortness,
   15→48prototype deployment shift and304561predicted IDs differ. Their
   numeric differences are descriptive, not an isolated frontend causal
   experiment. Selected Train mean available observations is
   {shift['Train_selected']['mean_available_observations']:.4f} vs Native
   {shift['Native_all']['mean_available_observations']:.4f}; match-conditioned
   lengths and Known prototype/physical coverage are in the actual posthoc
   descriptive diagnostic, not causal interventions. No GT repair or
   counterfactual retraining was performed.
9. **Physical restriction?** Known reliable1400/4413,Novel189/819. Main fixed
   reuse527 has118reliable current opportunities,409missed; only87have an
   earlier reliable same-category match. Other orders are in the support
   table. Strict both>=16support averages13.17%, **not** an absolute ceiling
   for true clipped short-track replay; any-observation support is larger.
   Unknown membership and semantic errors can reduce purity further.
10. **Enough for a new-method TrackOCD paper?** No supported core novelty
   claim currently: unseen geometry, independent temporal benefit and safe
   persistence are not established in controlled tests; Native/foundation
   provenance/coverage remains incomplete. This is a reproducible research
   negative result, not a publication-success claim or permission for rescue
   tuning. Any future loss, data or frontend intervention needs a new explicit
   hypothesis/authorization; no automatic extra experiment is launched.

## Full Val fixed p16 comparison sign counts

```json
{json.dumps(stable,indent=2)}
```

Three training seeds/four orders are not independent heldout categories.
Conditional500video bootstrap fixes global mapping and memory stream; it is
not a rerun/causal CI. Mean H averages actual H, never H of mean Old/New.
Unmatched does not mean proven background/pollution; unknown cannot certify
pure Novel. PHE INCOMPARABLE and historical M1 blocked are retained.
'''
    atomic_write_text(docs/'SCIENTIFIC_CONCLUSION.md',questions)
    support_table='| Order | fixed GT reuse | current reliable | missed current | both any observed | both>=16 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in physical_rows:support_table+='| '+' | '.join(str(v) for v in r.values())+' |\n'
    report=f'''# TrackOCD Train-first final report

Completed real33fits / 99retained checkpoints / 33selected checkpoints;
420representation +1800policy controlled cases and840full limited-MASA
executions /2040explicit logical aliases. Outcome: **credible scientific
negative**, not failed implementation, strongfrontend PASS or new-method
paper proof. This is the user-authorized independent main, not R2. Historical
M1 stays BLOCKED_FRONTEND_QUALITY; no historical outcomes/assets overwritten.

## Legal data and training

TAO Train inherited Known only,500videos/18274images inventory2196tracks /
48supportedcategories. Eligible fittingpool15classes1305tracks14714available
observations, not 'all48 optimized' orall1305visited ineachfit. Full1000-step
class-balanced runs visit933/938/948distincttracks for the three seeds,
identical acrosspairedmodels. These arefullfit counts,not the separately
unrecorded footprint ofearlier250/500selectedcheckpoints. All5partitions
video-disjoint; development/policy/final
pseudo category reservations disjoint from fitting and each other. 30tracks
without a lawful selected role excluded explicitly,1026short inventorytracks
retained eligible. Commonselected2166tracks/24628observations;151prototype
tracks/48categories,258dev/229policy/223heldout(207Known+16pseudo). Features,
weights,rawGT/predNPZ/privateflags/ledgers never uploaded toGitHub.

FrozenDINOv2 ViTB14/518bilinear/context.1,commonFP16,cpuFP32coreforward.
T1nine1000-step fits weights0/1/5,globaldevselectplainA1,selected500/250/500;
T2sixmatched1000-step temporal/staticcapacity fits withsameA1selectedsteps;
T3eighteen20epoch×4order actualpredicted-memoryfits,708params D1/D2,
sameinit/data/budget,frozenrepresentation,noGTrepair. Full reallosses and
pairedinit/batch/checkpointhashes are delivered. No Val Novel fitting/tuning,
Novel text input,Test access,newassets or physicalrerun.

## Controlled Train p16 results

Percentages: four-ordermean per seed then three-seed mean,rawonlyonecontrol.

{table(train_rows,'Method')}

Representation raw geometry and role-separated retrieval are in
TRAIN_FIRST_REPRESENTATION_ROLE_ANALYSIS.json. 1020existing sealedcase metrics
exactly rechecked,480pairederror comparisons; fixes AND introducederrors
reported. All2220 controlled rows are retained alongsideT5 in canonicalCSVs.

## Frozen full limited-MASA p16 results

All988videos/36375annotatedframes/304561IDs including114380singletons,
1294110commonfirst-at-most16observations. MainGT4413Known/819Novel and
fixed527/529/547/531reuse,notmatched-only1400/189denominators. Short/unmatched
IDs mutate memory whencommitted. Target1.0 is a Train-dev operatingpoint,
not Valcoveragepromise; actualcoverage in the tables/pairedresults.
GT unresolved/WAIT-rate metrics include physical misses; distinguish explicit
predicted-ID WAIT action rate (runtime), which has a different denominator.

{table(val_rows,'Method')}

HOTA {val['physical_native_reference']['HOTA']:.10f}, AssA {val['physical_native_reference']['AssA']:.10f},
DetA {val['physical_native_reference']['DetA']:.10f}, DetRe {val['physical_native_reference']['DetRe']:.10f}
are the **same existing canonical Native physicalreference** for everymethod,
not new tracking runs orsemanticphysicalgains. ANNOTATED-CADENCE CAUSAL REPLAY,
not dense-frameonline. PHE INCOMPARABLE; incomplete supervision provenance
precludes strict all-Novel-supervision-exclusion claims.

{support_table}

Posthoc descriptive Known prototype/physical-coverage counts (not new GT input
to frozen inference and not a preregistered causal intervention):

```json
{json.dumps(shift['Known_GT_prototype_and_physical_coverage'],indent=2)}
```

Strict>=16support13.17% isnotclippedshort-trackabsoluteceiling. See
SCIENTIFIC_CONCLUSION.md for allten requiredquestions, stablepair signs,
failureinterpretations and limitations. No post-result model/bias/cost tuning.

## Reproduction, checkpoints and provenance

Branch `codex/trackocd-core-training-limited-masa`; report source registered
and exactremoteverified at `{commit}`. Original frozen inference registration:
`{', '.join(lineage['original_frozen_inference_registration_commits'])}`;
scheduling-only recovery registration:
`{lineage['scheduling_recovery_registration_commit'] or 'not applicable'}`;
actual evaluator case registrations:
`{', '.join(lineage['actual_case_evaluator_registration_commits'])}`;
full evaluator collector registration `{val['preregistration_commit']}`.
Recovery did not create new models, points or a new method after Val metrics.
Both actual successful parent terminal receipts are required; adopted orphan
exit codes are unavailable and never fabricated as zero. Their atomic group
completion and every immutable case seal constitute explicit terminal proof.
Model freeze `{result['model_freeze_sha256']}`
binds52protectedsource/config/receipt/checkpointhashes. Finalreportcommit and
exactremote are verified **after** committing these reports; no impossible
self-referential commit SHA is fabricated inside its own artifact.

Use the existing `/data3/liuyeqiang/.venvs/trackocd-a100` entry environment;
configs, script CLI and actual receipts specify input/source/protocol hashes.
Train: `train_core_representation.py` with `--phase representation` or
`--phase evidence`, and `train_persistent_policy.py`; controlled:
`evaluate_core_representation.py` and `evaluate_persistent_policy.py`.
Predicted: `extract_limited_masa_features.py`,
`validate_full_limited_masa_features.py`, `run_limited_masa_inference.py`,
`evaluate_limited_masa.py`; require exact prereg remote SHA and actualcompleted
inputs, not historical PID/session reuse. Existing completed outputs refuse
overwrite. For an independent reproduction use a separately preregistered
empty destination/config identity and lawful matching localassets, never
delete/reset originaloutputs. Exhaustive CPU/CUDA actual40case proofs and
syntheticregressions accompany the executor; no historicalPython3.7 equality
or tiny-proof CUDA speedup is inferred.

{checkpoint_table}

## Actual resource receipts

```json
{json.dumps(resource,indent=2)}
```

Full resource/source/artifact SHA256 maps and99localcheckpoint identities are
in CORE_TRAINING_FINAL_RESULT.json. Preserve25%systemRAM,soft15/hard30GiB new
storage,4GiB inferenceworker peakguard,4GiB GPUplan+8GiBfree reserve. Owned
workers only; no externalprocess termination/deletion/newdownload. Rawprivate
assets staylocal. Trainingloss completion and349+unitregressions are engineering
evidence,not scientific PASS. No further safe-feedback/training experiment is
automatically authorized by this result.
'''
    atomic_write_text(docs/'CORE_TRAINING_FINAL_REPORT.md',report)
    old=old.replace(RUNNING_PROTOCOL_STATUS,
        'All common features,840fullstream semantic executions and2040explicit logical\nalias reports are complete. The following frozen protocol was registered before\nthose metrics. Actual results and comparison flags are appended below.')
    atomic_write_text(docs/'LIMITED_MASA_EVALUATION.md',old+'\n## Completed actual full semantic evaluation\n\n'+table(val_rows,'Method atp16 Train-devtarget1 or rawrepresentation')+'\n'+support_table+'\nSee full per-order/seed/prefix/coverage JSON andCSV, explicit alias manifest,\nmatchedcoverage flags andintroduced-error counts. Samephysicalreference for all.\n')
    result['final_document_sha256']={n:sha256_file(docs/n) for n in ('CORE_TRAINING_FINAL_REPORT.md','METHOD_ARCHITECTURE.md','TRAINING_PROTOCOL.md','LIMITED_MASA_EVALUATION.md','SCIENTIFIC_CONCLUSION.md')}
    atomic_json(out/'CORE_TRAINING_FINAL_RESULT.json',result)
    print(json.dumps({'status':result['status'],'actual_fits':33,'limited_unique':840,'limited_logical':2040,'conservative_private_file_bytes':local_bytes}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);a=p.parse_args();main(a.preregistration_commit)
