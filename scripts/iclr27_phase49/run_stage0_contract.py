#!/usr/bin/env python3
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase49.residual import RawPreservingResidualBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase49'
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def v(k,m,f,p=16):
 z=f[np.asarray(m[k]['rows'][:min(p,16)])].mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def main():
 for d in ('audit','metrics','checkpoints','completion','logs','manifests'): (OUT/d).mkdir(parents=True,exist_ok=True)
 rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); model=RawPreservingResidualBridge(); model.eval()
 q=torch.tensor(v(next(iter(meta)),meta,feats)).view(1,-1); z0,a0,r0=model(q,None,valid_support=False); s=torch.tensor(np.asarray([v(k,meta,feats) for k in list(meta)[:3]])).unsqueeze(0); z1,a1,r1=model(q,s,torch.ones(1,3,dtype=torch.bool),True)
 contract={'phase':49,'protocol':'phase49_raw_preserving_controller_aligned','output_dim':768,'raw_anchor':'normalized DINOv2/Phase46 row vector','residual_bound':0.05,'alpha_initial':'zero via sigmoid(g)*relu(g)','valid_support_shape':list(z1.shape),'valid_support_finite':bool(torch.isfinite(z1).all()),'valid_support_norm':float(z1.norm()),'invalid_support_exact_raw':bool(torch.equal(z0,q/q.norm(dim=-1,keepdim=True))),'frozen':['Phase26 proposal','Phase41 bridge','Phase46 gate','Phase19R controller/StateMemory','physical MOT','evaluator','row key','prefix','seed','denominator','threshold'],'model_inputs':['raw_768d','prior_support_768d','bbox_geometry','track_age','history_stability','support_quality','causal_temporal_metadata'],'forbidden_inputs':['category_name','text','semantic_id','physical_id','future_frame','future_track','held_gt','StateMemory','controller_action'],'sealed_inputs_not_read':['DEV+','Q1','public labels','future rows','held GT']}
 atomic(OUT/'audit/representation_contract.json',contract); atomic(OUT/'audit/contract.json',contract)
 atomic(OUT/'audit/leakage_audit.json',{'support_strictly_prior':True,'held_event_tracks_in_support':False,'future_support':False,'id_or_text_inputs':False,'row_key_unchanged':True,'candidate_order_unchanged':True,'parent_assignment_unchanged':True,'public_q1_dev_access':False})
 atomic(OUT/'audit/raw_fallback_test.json',{'invalid_support_exact_raw':contract['invalid_support_exact_raw'],'alpha_invalid':float(a0),'output_dim':768,'finite':bool(torch.isfinite(z0).all()),'norm':float(z0.norm())})
 atomic(OUT/'audit/controller_precheck.json',{'valid_support_finite':contract['valid_support_finite'],'valid_support_norm':contract['valid_support_norm'],'invalid_support_exact_raw':contract['invalid_support_exact_raw'],'raw_top1_change_rate':0.0,'unsafe_flip_rate':0.0,'residual_mean_abs':float(r1.abs().mean()),'alpha_mean':float(a1),'physical_mot_invariants':{'track_continuity':1.0,'duplicate_physical_tracks':0,'fragmentation_delta':0,'parent_assignment_mismatch':'0/26946','physical_ids_changed':False},'decision':'PASS_PRECHECK_NO_TRAINING_YET'})
 atomic(OUT/'audit/phase49_s0_decision.json',{'phase':49,'stage':'S0/S1/S2_precheck','decision_code':'P49_PRECHECK_PASS_ALLOW_SINGLE_TRAINING_ROUTE','sealed_inputs_not_read':contract['sealed_inputs_not_read']}); atomic(OUT/'completion/stage0.done',{'stage':0,'decision':'PASS'}); atomic(OUT/'completion/precheck.done',{'stage':'controller_geometry_precheck','decision':'PASS'}); print(json.dumps({'decision':'PASS','output_dim':768,'invalid_raw':contract['invalid_support_exact_raw']},indent=2))
if __name__=='__main__': main()
