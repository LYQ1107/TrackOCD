"""Validate four-fold main completion without changing any metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("outputs/iclr27_phase19r/audit/main_artifact_validation.json")); a = p.parse_args()
    rows = []; passed = True
    for fold in range(4):
        done = Path(f"outputs/iclr27_phase19r/completion/fold{fold}.done"); summary = Path(f"outputs/iclr27_phase19r/metrics/fold{fold}_training.json")
        ok = done.exists() and summary.exists(); rec = {}
        if ok:
            rec = json.loads(summary.read_text()); ok = bool(rec.get("full_registered_run")) and int(rec.get("updates", 0)) >= 50000 and int(rec.get("finite_updates", 0)) == int(rec.get("updates", 0))
            vals = [x.get("validation", {}) for x in rec.get("logs", [])]
            ok = ok and all(v.get("selection_metric_source") == "full_persistent_held_known_event_evaluator" for v in vals)
            ok = ok and all(math.isfinite(float(v.get("selection_score", 0.0))) for v in vals)
        passed = passed and ok; rows.append({"fold": fold, "passed": ok, "done": done.exists(), "summary": summary.exists(), "updates": rec.get("updates"), "finite_updates": rec.get("finite_updates"), "best_step": rec.get("best_step"), "selection_metric_source": rec.get("selection_metric_source")})
    result = {"protocol": "trackocd_iclr27_phase19r_main_artifact_validation", "passed": passed, "folds": rows}
    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
