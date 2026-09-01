"""Select the single preregistered fallback using held-known results only."""
from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path
from typing import Any


OUT = Path("outputs/iclr27_phase19r")


def read_fold(fold: int) -> dict[str, Any]:
    p = OUT / "metrics" / f"fold{fold}_training.json"
    if not p.exists(): return {"missing": True, "fold": fold}
    return json.loads(p.read_text())


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=OUT / "manifests/fallback_selection.json"); a = p.parse_args()
    folds = [read_fold(f) for f in range(4)]; checks = []; diagnostics = []
    for rec in folds:
        if rec.get("missing"):
            checks.append(False); diagnostics.append({"fold": rec["fold"], "reason": "missing training summary"}); continue
        logs = rec.get("logs", []); val = logs[-1].get("validation", {}) if logs else {}
        em = val.get("persistent_event_metrics", {})
        fm = json.loads((OUT / "manifests/fold_manifest.json").read_text())
        held_count = len(fm["folds"][int(rec["fold"])] .get("held_categories", []))
        gate = {"existing_precision": float(em.get("existing_precision", 0.0)) >= .70,
                "negative_false_merge": float(em.get("negative_false_merge_rate", 1.0)) <= .15,
                "category_coverage": int(em.get("category_coverage", 0)) >= max(1, math.ceil(.75 * held_count)),
                "known_micro": float(val.get("known_micro", 0.0)) >= .60,
                "known_macro": float(val.get("known_macro", 0.0)) >= .50}
        checks.append(all(gate.values())); diagnostics.append({"fold": rec["fold"], "step": rec.get("best_step"), "gate": gate,
                                                                  "persistent_event_metrics": em, "episode_known_micro": val.get("known_micro"),
                                                                  "episode_known_macro": val.get("known_macro")})
    main_pass = bool(all(checks))
    result = {"protocol": "trackocd_iclr27_phase19r_fallback_selection", "main_internal_gate_pass": main_pass,
              "selected": None if main_pass else "fallback_f_a_gaussian",
              "selection_before_public_truth_join": True,
              "public_truth_joined": False,
              "fallback_spec": "PCA + Ledoit-Wolf covariance + causal Gaussian state scoring",
              "fold_diagnostics": diagnostics}
    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, out)
    print(json.dumps({"main_internal_gate_pass": main_pass, "selected": result["selected"]}, sort_keys=True))


if __name__ == "__main__": main()
