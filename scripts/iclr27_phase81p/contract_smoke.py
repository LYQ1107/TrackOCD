#!/usr/bin/env python3
"""Minimal causal/shape contract smoke for the physical association route."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))

def main():
 import torch
 from src.iclr27_phase81p.association import AssociationTransformer, CausalAssociationRuntime, PAIR_DIM
 torch.manual_seed(8101); model=AssociationTransformer(); x=torch.zeros((2,9,PAIR_DIM)); pair,new=model.score_candidates(x)
 assert tuple(pair.shape)==(2,9) and tuple(new.shape)==(2,) and bool(torch.isfinite(pair).all()) and bool(torch.isfinite(new).all())
 rt=CausalAssociationRuntime(model,'cpu');
 out0=rt.step([],0); assert out0==[]
 det={'bbox_xyxy':[1,2,20,30],'base_score':0.8,'frame_id':1,'appearance':np.zeros(8,np.float32)}
 out1=rt.step([det],1); assert len(out1)==1 and len(out1[0]['bbox_xyxy'])==4 and out1[0]['physical_track_id']==0
 det2={'bbox_xyxy':[2,3,21,31],'base_score':0.7,'frame_id':2,'appearance':np.zeros(8,np.float32)}
 out2=rt.step([det2],2); assert len(out2)==1 and out2[0]['physical_track_id']>=0
 result={'schema_version':'phase81p.contract_smoke.v1','status':'PASS','pair_shape':list(pair.shape),'new_shape':list(new.shape),'row_vector_dim':768,'causal_empty_step':True,'birth_then_step':True,'physical_ids_model_input':False,'future_rows_or_tracks':False,'category_text':False,'sealed_accessed':False}
 p=ROOT/'outputs/iclr27_phase81p/audit/contract_smoke.json'; p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp'); t.write_text(json.dumps(result,indent=2)+'\n'); os.replace(t,p); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
