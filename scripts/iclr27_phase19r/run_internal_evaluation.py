"""Run persistent held-known internal metrics for all repaired candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cpu"); p.add_argument("--out",type=Path,required=True); p.add_argument("--candidate",choices=["raw","age","talon","main","fallback"],default="main"); a=p.parse_args(); dev=torch.device(a.device); rows=[]
    for fold in range(4):
        data=Phase19RData(fold)
        if a.candidate in {"raw","age","talon"}: rows.append(evaluate_candidate(a.candidate,data,None,dev))
        elif a.candidate == "main": rows.append(evaluate_candidate("main",data,Path(f"outputs/iclr27_phase19r/checkpoints/fold{fold}_best_internal.pt"),dev))
        else:
            # Fallback is selected only after the main gate and uses the
            # pre-registered Gaussian persistent controller, never the main
            # checkpoint under a relabeled candidate name.
            rows.append(evaluate_candidate("fallback", data, None, dev))
    result={"protocol":"trackocd_iclr27_phase19r_internal_evaluation","candidate":a.candidate,"folds":rows,"category_macro_mean":sum(x["metrics"]["category_macro_reuse"] for x in rows)/len(rows),"existing_precision_mean":sum(x["metrics"]["existing_precision"] for x in rows)/len(rows),"false_merge_mean":sum(x["metrics"]["negative_false_merge_rate"] for x in rows)/len(rows),"known_safety_not_in_event_evaluator":True}; a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps({"candidate":a.candidate,"fold_metrics":[x["metrics"] for x in rows]},indent=2,sort_keys=True))


if __name__ == "__main__": main()
