"""Re-evaluate historical Phase19 checkpoints with persistent memory.

The old JSONs are never overwritten; this script quantifies the E4 reset
effect using the old model and corrected stream boundary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.iclr27_phase19.data.stream import Phase19Data
from src.iclr27_phase19.evaluation.evaluate import ModelController
from src.iclr27_phase19.runtime.state_machine import CausalStateMachine


def one(fold: int, name: str, checkpoint: Path, device: torch.device, ladder: str) -> dict:
    data = Phase19Data(fold); c = ModelController(data, checkpoint, device, deferred=False)
    def include(k: str) -> bool:
        if ladder == "L2": return True
        q = [data.prefix(k, p)[2] for p in range(len(data.track_rows[k]))]
        if ladder == "L0": return any(data.rows[i]["assigned"] == "1" and float(data.rows[i]["row_iou"]) >= .5 for i in data.track_rows[k])
        return float(np.mean(q)) >= .35
    hv = set(int(x) for x in data.fold_record["validation_videos"]); held = [k for k in data.track_rows if include(k) and data.track_cat_eval[k] in data.held_categories and data.track_video[k] in hv]; vis = [k for k in data.track_rows if include(k) and data.track_cat_eval[k] in data.supported_set and data.track_cat_eval[k] not in data.held_categories and data.track_video[k] in hv]
    orders = []; rng = np.random.default_rng(1901 + fold)
    for order in range(3):
        tracks = held + vis; rng.shuffle(tracks); sm = CausalStateMachine(c.model, len(data.supported_ids), max_states=8, allow_defer=False); pred=[]; gt=[]; npred=[]; ngt=[]
        for key in tracks:
            cat=data.track_cat_eval[key]
            for pos in range(len(data.track_rows[key])):
                raw, _, _, _ = data.prefix(key,pos); geom=torch.from_numpy(data.prefix(key,pos)[1]).to(device); row=data.rows[data.track_rows[key][pos]]
                got=sm.predict(torch.from_numpy(raw).to(device),geom,int(row["video_id"]),key)
                if got["semantic_id"] is not None:
                    for st in sm.states:
                        if st.sid == got["semantic_id"] and st.oracle_category is None: st.oracle_category=cat
                sid = data.supported_ids[int(got["semantic_id"])] if got["action"] == "KNOWN" and got["semantic_id"] is not None else got["semantic_id"]
            pred.append(-1 if sid is None else sid); gt.append(cat)
            if cat in data.held_categories: npred.append(-1 if sid is None else sid); ngt.append(cat)
        pu=sorted(set(pred)); gu=sorted(set(gt)); mat=np.zeros((len(pu),len(gu)),int)
        for p,g in zip(pred,gt): mat[pu.index(p),gu.index(g)] += 1
        rr,cc=linear_sum_assignment(-mat) if mat.size else ([],[]); mapping={pu[int(r)]:gu[int(c)] for r,c in zip(rr,cc)}
        orders.append({"order": order, "all_accuracy": float(mat[rr,cc].sum()/max(len(gt),1)) if len(rr) else 0., "old_accuracy": float(np.mean([p==g for p,g in zip(pred,gt) if g not in data.held_categories])) if vis else 0., "new_hungarian_accuracy": float(np.mean([mapping.get(p, object()) == g for p,g in zip(npred,ngt)])) if npred else 0., "nmi_novel": float(normalized_mutual_info_score(ngt,npred)) if len(set(ngt))>1 else 0., "ari_novel": float(adjusted_rand_score(ngt,npred)) if len(ngt)>1 else 0., "novel_discovery_count_error": abs(len(set(npred))-len(set(ngt))), "state_count_end": sm.states.__len__()})
    return {"candidate": name, "fold": fold, "ladder": ladder, "persistent_memory": True, "orders": orders, "mean": {k: float(np.mean([o[k] for o in orders])) for k in orders[0] if k not in {"order","state_count_end"}}, "order_sensitivity": float(np.std([o["all_accuracy"] for o in orders]))}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cpu"); p.add_argument("--ladder",choices=["L0","L1","L2"],default="L2"); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); root=Path("outputs/iclr27_phase19/checkpoints"); rows=[]
    for name in ["main","fallback_a"]:
        for fold in range(4): rows.append(one(fold,name,root/f"{name}_fold{fold}_best.pt",torch.device(a.device),a.ladder))
    result={"protocol":"trackocd_iclr27_phase19r_old_checkpoint_persistent_evaluation","ladder":a.ladder,"rows":rows,"old_reset_metrics_retained":"outputs/iclr27_phase19/metrics/*_L2.json","reset_bug_repaired":True}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__": main()
