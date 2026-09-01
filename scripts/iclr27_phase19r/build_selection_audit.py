"""Collect the exact fixed-event checkpoint-selection traces."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("outputs/iclr27_phase19r/audit/selection_audit.json")); a = p.parse_args()
    folds = []
    for fold in range(4):
        path = Path(f"outputs/iclr27_phase19r/metrics/fold{fold}_training.json")
        if not path.exists():
            folds.append({"fold": fold, "missing": True}); continue
        summary = json.loads(path.read_text())
        folds.append({"fold": fold, "updates": summary.get("updates"), "best_step": summary.get("best_step"),
                      "best_internal_score": summary.get("best_internal_score"),
                      "selection_metric_source": summary.get("selection_metric_source"),
                      "fixed_persistent_events": summary.get("fixed_persistent_events"),
                      "validation_points": [{"step": x.get("step"), "selection_score": x.get("validation", {}).get("selection_score"),
                                             "episode_proxy_selection_score": x.get("validation", {}).get("episode_proxy_selection_score"),
                                             "selection_metric_source": x.get("validation", {}).get("selection_metric_source"),
                                             "persistent_event_metrics": x.get("validation", {}).get("persistent_event_metrics", {})}
                                            for x in summary.get("logs", [])]})
    result = {"protocol": "trackocd_iclr27_phase19r_checkpoint_selection_audit", "selection_rule": "persistent_selection_score(event_metrics)",
              "folds": folds, "public_truth_joined": False}
    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, out)
    print(json.dumps({"folds": len(folds), "missing": [x["fold"] for x in folds if x.get("missing")]}, sort_keys=True))


if __name__ == "__main__": main()
