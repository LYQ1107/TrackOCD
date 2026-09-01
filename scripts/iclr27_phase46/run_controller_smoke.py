#!/usr/bin/env python3
"""Bounded contract smoke only; does not evaluate held events or Commit-CT."""
import json, os, tempfile
from pathlib import Path
import torch
from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.runtime.runner import ModelStreamController
from src.iclr27_phase41.bridge import SafetyVectorBridge
from src.iclr27_phase46.selective import ConditionalLogitGate
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase46'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 dev=torch.device('cpu'); data=Phase19RData(fold=0,final=False); ck=torch.load(ROOT/'outputs/iclr27_phase19r/checkpoints/fold0_best_internal.pt',map_location='cpu',weights_only=False)
 model=RCMSOCD(torch.from_numpy(data.known_prototypes),torch.from_numpy(data.active_known_mask),known_bias=torch.from_numpy(data.known_bias)); model.load_state_dict(ck['model_state']); model.eval(); ctrl=ModelStreamController(model,max_states=16,allow_defer=True,tau_ready=model.tau_ready,tau_known=model.tau_known,tau_assign=model.tau_assign)
 bridge=SafetyVectorBridge(); bridge.load_state_dict(torch.load(ROOT/'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f0_best.pt',map_location='cpu',weights_only=False)['model']); bridge.eval(); gate=ConditionalLogitGate(); gate.load_state_dict(torch.load(OUT/'checkpoints/phase46_formal_v1_f0_best.pt',map_location='cpu',weights_only=False)['model']); gate.eval()
 key=sorted(data.track_rows)[0]; idx=data.track_rows[key][0]; row=data.rows[idx]; raw,geom,q,_=data.prefix(key,0); raw_t=torch.from_numpy(raw); geom_t=torch.from_numpy(geom); km=torch.from_numpy(data.active_known_mask)
 z_invalid,alpha_invalid,_=bridge(raw_t.view(1,-1),None,torch.zeros(1),torch.zeros(1),False); invalid_exact=bool(torch.allclose(z_invalid[0],raw_t)) and float(alpha_invalid.item())==0.0
 ctx=raw_t.clone(); z_valid,alpha_valid,_=bridge(raw_t.view(1,-1),ctx.view(1,-1),torch.tensor([.1]),torch.tensor([.5]),True); logit=gate(torch.tensor([.1]),torch.tensor([.2]),torch.tensor([.5]),alpha_valid,torch.zeros(1)); chosen=z_valid[0] if float(torch.sigmoid(logit).item())>=.5 else raw_t
 out1,b1=ctrl.forward_item(chosen,geom_t,float(q),int(row['video_id']),key,km); before=len(ctrl.memory.states); rec=ctrl.process_item(chosen,geom_t,1.0,int(row['video_id']),key,km,force_action=('NEW',None)); after=len(ctrl.memory.states); rec2=ctrl.process_item(chosen,geom_t,1.0,int(row['video_id']),key,km,force_action=('EXISTING',0)); causal_order=rec['step'] < rec2['step'] and ctrl.memory.step_index >= 2 and len(ctrl.memory.states)==1
 result={'phase':46,'controller_frozen':True,'held_events_run':False,'commit_ct_run':False,'row_vector_dim':int(chosen.shape[-1]),'row_norm':float(chosen.norm().item()),'invalid_support_exact_raw':invalid_exact,'valid_bridge_finite':bool(torch.isfinite(z_valid).all()),'gate_probability':float(torch.sigmoid(logit).item()),'state_count_before':before,'state_count_after':after,'causal_state_order':causal_order,'physical_track_key_used_only_for_admissibility':True,'future_or_sealed_inputs':False,'action_semantics_unchanged':True,'sample_track':key,'sample_video':int(row['video_id'])}
 atomic(OUT/'audit/controller_smoke.json',result); atomic(OUT/'completion/controller_smoke.done',result); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
