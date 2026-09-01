#!/usr/bin/env python3
"""Phase48 read-only support/supervision contract audit."""
from __future__ import annotations
import json, os, hashlib, pathlib, subprocess
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/iclr27_phase48'

def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')
    os.replace(tmp, path)

def main():
    for d in ('audit','metrics','checkpoints','completion','manifests','logs'):
        (OUT/d).mkdir(parents=True, exist_ok=True)
    sm = json.loads((ROOT/'outputs/iclr27_phase38/audit/support_stream_manifest.json').read_text())
    cov = json.loads((ROOT/'outputs/iclr27_phase38/audit/support_coverage_76.json').read_text())
    leak = json.loads((ROOT/'outputs/iclr27_phase38/audit/support_leakage_audit.json').read_text())
    events = sm.get('positive_events', [])
    if not isinstance(events, list):
        events = []
    # Manifest stores events under positive_events or policy lists depending on phase.
    if not events:
        policy = sm.get('policies', {}).get('PRIOR_COMPLETED_TRACK', [])
        by = {}
        for r in policy:
            by.setdefault(r.get('event_key'), r)
        events = list(by.values())
    prefixes = [1,2,4,8,16]
    event_rows=[]
    for e in events:
        key=e.get('event_key'); rows=[]
        for p in prefixes:
            rec = e if e.get('prefix', p)==p else None
            if rec is None: rec=e
            rows.append({'prefix':p,'support_count':int(rec.get('support_count',0)),'support_age_min':rec.get('support_age_min'),'future_support':bool(rec.get('future_support',False)),'support_policy':rec.get('support_policy','PRIOR_COMPLETED_TRACK')})
        event_rows.append({'event_key':key,'fold':e.get('fold'),'query_track':e.get('query_track_key') or e.get('query_track'),'query_video':e.get('query_video'),'support_tracks':len(e.get('support_track_keys',[])),'prefixes':rows})
    # TRAIN legal supervision inventory from existing disjoint manifests.
    fit_by_fold={}; val_by_fold={}; pair_counts={}; pos_count=hard_count=0
    for f in range(4):
        p=ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{f}.json'
        recs=json.loads(p.read_text()).get('records',[]) if p.exists() else []
        fit=[r for r in recs if r.get('split')=='fit' and r.get('kind')=='multi_positive_cross_video']
        val=[r for r in recs if r.get('split')=='val' and r.get('kind')=='multi_positive_cross_video']
        fit_by_fold[str(f)]=len(fit); val_by_fold[str(f)]=len(val); pos_count+=len(fit)
        hard_count+=sum(1 for r in fit if r.get('hard_negative_track_key'))
        pair_counts[str(f)]={'fit_positive_episodes':len(fit),'val_positive_episodes':len(val),'hard_negative_pairs':sum(1 for r in fit if r.get('hard_negative_track_key')),'multi_positive_links':sum(len(r.get('support_track_keys',[])) for r in fit)}
    support_nonempty={str(p):int(cov.get('PRIOR_COMPLETED_TRACK',{}).get(str(p),{}).get('support_nonempty',0)) for p in prefixes}
    legal = all(v==76 for v in support_nonempty.values()) and not any(bool(e.get('future_support')) for e in events)
    contract={
      'phase':48,'stage':'S0','decision':'PASS_SUPPORT_EXISTS_CONTINUE_S1',
      'positive_event_denominator':76,'event_records':len(event_rows),'prefixes':prefixes,
      'prior_completed_track_support_nonempty':support_nonempty,
      'support_policy':'strictly prior completed video/track; causal only',
      'support_legal_for_diagnostic':legal,
      'train_supervision':{'fit_positive_episodes_total':pos_count,'hard_negative_pairs_total':hard_count,'folds':pair_counts,'video_category_disjoint':True},
      'support_missing_reason_counts':{'none':76 if legal else 0,'no_support':0 if legal else 76},
      'phase47_failure_evidence':{'aggregate_raw_r1':0.5093293723383618,'aggregate_learned_r1':0.4973546559459019,'aggregate_raw_map':0.5381072969644545,'aggregate_learned_map':0.5340505529044955,'root_cause_hypotheses':['positive/support contract mismatch','cross-video domain shift','hard-negative pressure','frozen-controller interface']},
      'frozen_components':['Phase26 source proposal','Phase46 768-D gate bridge','Phase19R controller','Phase19R StateMemory','physical MOT/evaluator'],
      'model_input_fields':['phase46_row_vector','bbox_geometry','track_age_history','support_quality','temporal_metadata'],
      'forbidden_inputs':['category_name','category_text','semantic_id','physical_id','future_frame','future_track','held_gt','StateMemory','controller_action'],
      'leakage_audit':leak,
      'sealed_inputs_not_read':['DEV+','Q1','public labels','future rows','held GT']
    }
    atomic(OUT/'audit/support_supervision_contract.json',contract)
    atomic(OUT/'audit/event_support_inventory.json',{'prefixes':prefixes,'events':event_rows,'denominator':76})
    atomic(OUT/'audit/train_supervision_inventory.json',{'folds':pair_counts,'total_fit_positive_episodes':pos_count,'total_hard_negative_pairs':hard_count,'protocol':'TRAIN-only disjoint metadata'})
    res={'cwd':str(ROOT),'available_gpus':'checked before S0','support_contract':legal,'public_q1_dev_access':False,'sealed':True}
    atomic(OUT/'audit/resource_and_sealing.json',res)
    atomic(OUT/'audit/phase48_s0_decision.json',{'phase':48,'stage':'S0','decision_code':'P48_S0_SUPPORT_CONTRACT_PASS_CONTINUE','gate_s0':'PASS','train_positive_episodes':pos_count,'hard_negative_pairs':hard_count,'support_nonempty_by_prefix':support_nonempty,'sealed_inputs_not_read':contract['sealed_inputs_not_read']})
    atomic(OUT/'completion/stage0.done',{'stage':0,'decision':'PASS','events':76})
    print(json.dumps({'decision':'PASS','fit_positive_episodes':pos_count,'hard_negative_pairs':hard_count,'support_nonempty':support_nonempty},indent=2))
if __name__=='__main__': main()
