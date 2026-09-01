"""Run the one preregistered Gaussian fallback on all four held-known folds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("outputs/iclr27_phase19r/metrics/fallback_full_internal.json")); p.add_argument("--device", default="cpu"); a = p.parse_args()
    rows = []
    for fold in range(4):
        rows.append(evaluate_candidate("fallback", Phase19RData(fold), None, torch.device(a.device)))
    result = {"protocol": "trackocd_iclr27_phase19r_fallback_f_a_full_four_fold", "candidate": "fallback_f_a_gaussian", "folds": rows,
              "category_macro_mean": sum(x["metrics"]["category_macro_reuse"] for x in rows) / 4.0,
              "existing_precision_mean": sum(x["metrics"]["existing_precision"] for x in rows) / 4.0,
              "false_merge_mean": sum(x["metrics"]["negative_false_merge_rate"] for x in rows) / 4.0,
              "public_truth_joined": False, "selection_manifest": "outputs/iclr27_phase19r/manifests/fallback_selection.json"}
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_name(a.out.name + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, a.out)
    done = Path("outputs/iclr27_phase19r/completion/fallback_full.done"); done.parent.mkdir(parents=True, exist_ok=True); dtmp = done.with_name(done.name + ".tmp"); dtmp.write_text("done\n"); os.replace(dtmp, done)
    print(json.dumps({"complete": True, "category_macro_mean": result["category_macro_mean"], "false_merge_mean": result["false_merge_mean"]}, sort_keys=True))


if __name__ == "__main__": main()
