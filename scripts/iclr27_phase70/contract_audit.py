#!/usr/bin/env python3
"""Phase70 frozen-interface audit (no training and no public/sealed access)."""
from __future__ import annotations
import csv, hashlib, json, os, pathlib, subprocess, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/iclr27_phase70/audit/phase70_contract.json'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,t=tempfile.mkstemp(prefix='.'+p.name,dir=str(p.parent))
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(t,p)
    finally:
        if os.path.exists(t): os.unlink(t)

def main():
    ck=[]
    for f in range(4):
        p=ROOT/f'outputs/iclr27_phase69/checkpoints/fold{f}_repair1/checkpoint.pth'
        ck.append({'fold':f,'path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else 0,'sha256':sha(p) if p.exists() else None})
    events=[]
    for name in ('held_known_positive_events.jsonl','held_known_negative_events.jsonl'):
        p=ROOT/'outputs/iclr27_phase19r/manifests'/name
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        events.append({'file':str(p),'sha256':sha(p),'count':len(rows),'fold_counts':{str(f):sum(int(x.get('fold',-1))==f for x in rows) for f in range(4)},'used_as_model_input':False})
    csv_path=ROOT/'data/iclr27_phase19r/sources/public_rows_corrected.csv'
    with csv_path.open(newline='') as f:
        rows=list(csv.DictReader(f))
    track_keys=sorted({f"v{int(r['video_id'])}:p{int(r['track_id'])}" for r in rows})
    out={
      'phase':70,
      'protocol':'trackocd_phase70_frozen_ovtr_mot_then_existing_dsct_ocd',
      'cwd':str(ROOT),
      'frozen_phase69_checkpoints':ck,
      'phase46_bridge_gate':{
        'bridge_namespace':'outputs/iclr27_phase41/checkpoints',
        'gate_namespace':'outputs/iclr27_phase46/checkpoints',
        'row_vector_dim':768,
        'raw_fallback':'exact on invalid/missing support',
        'frozen':True,
      },
      'phase19r_controller':{
        'checkpoint_pattern':str(ROOT/'outputs/iclr27_phase19r/checkpoints/fold{fold}_best_internal.pt'),
        'controller_code':str(ROOT/'src/iclr27_phase19r/runtime/runner.py'),
        'state_memory_unchanged':True,
        'threshold_sweep':False,
      },
      'events':events,
      'event_total_positive':sum(x['count'] for x in events if 'positive' in x['file']),
      'event_total_negative':sum(x['count'] for x in events if 'negative' in x['file']),
      'feature_rows':len(rows),
      'physical_track_keys':len(track_keys),
      'causal_prefixes':[1,2,4,8,16],
      'inference_inputs':['RGB/query representation','bbox geometry','motion/history','causal quality','support metadata'],
      'forbidden_inference_inputs':['category names/text','semantic IDs','physical IDs as features','future frames/tracks','held GT','DEV+','Q1/public new-model labels','controller action as representation input'],
      'sealed_public_q1_accessed':False,
      'held_gt_used_as_model_input':False,
      'physical_id_semantic_separation':True,
      'proposal_physical_tracker_frozen':True,
      'phase70_route':'reuse OVTR DSCT semantic/state heads; semantic stage b -> assign/create stage c -> one joint stage d; no new encoder/backbone',
    }
    atomic(OUT,out)
    print(json.dumps({'out':str(OUT),'positive':out['event_total_positive'],'negative':out['event_total_negative'],'tracks':len(track_keys)},indent=2))
if __name__=='__main__': main()
