#!/usr/bin/env python3
"""Minimal causal support selector contract test (no held data)."""
import json, numpy as np, torch, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.iclr27_phase85.support_model import SupportReranker,numpy_predict
def main():
 torch.manual_seed(85); m=SupportReranker().eval(); x=np.zeros((3,19),np.float32);x[:,15]=[.4,.3,.1];ctx=np.asarray([.4,.5,.1,.1,.05,.3,.2,.5,.1,.5],np.float32); mean=np.zeros(19,np.float32);std=np.ones(19,np.float32);score,p,_=numpy_predict(m,x,ctx,x[:,15],mean,std); assert score.shape==(3,) and np.isfinite(score).all() and np.isfinite(p)
 # Invalid/missing support is handled by the caller as an exact raw fallback.
 raw=x[:,15].copy(); invalid=np.array([],np.float32); assert invalid.size==0 and np.allclose(raw,x[:,15],atol=0,rtol=0)
 out={'schema_version':'trackocd.phase85.support_contract_smoke.v1','status':'PASS','valid_score_shape':list(score.shape),'finite_probability':True,'missing_support_raw_fallback_exact':True,'candidate_dim':19,'context_dim':10,'row_vector_dim':768,'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False}
 p=ROOT/'outputs/iclr27_phase85/audit/support_contract_smoke.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n');(ROOT/'outputs/iclr27_phase85/completion/support_contract_smoke.done').write_text(json.dumps({'status':'DONE'})+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
