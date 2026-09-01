"""Post-freeze known-audit measurement for the Phase 19 public contract.

This script is deliberately separate from training and candidate selection.  It
may read the evaluator-only supported-known labels only after the prediction
freeze marker exists.  The metric follows the Phase 18 audit convention: a
KNOWN decision updates the held semantic action, while DEFER/NEW/EXISTING do
not erase a previously committed known action.  The reported micro and
category-macro scores are therefore causal action safety, not an offline
closed-set oracle score.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase19.data.stream import Phase19Data
from src.iclr27_phase19.evaluation.evaluate import ModelController, RawController

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase19"


def groups(data: Phase19Data) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(data.rows):
        if row.get("role17") == "known_audit" and row.get("gt_role_common") == "supported_known":
            out[f"v{int(row['video_id'])}:p{int(row['track_id'])}"].append(i)
    for key in out:
        out[key].sort(key=lambda i: (int(data.rows[i]["event_rank"]), i))
    return dict(out)


def run_candidate(name: str, checkpoint: Path | None, device: torch.device) -> dict[str, Any]:
    data = Phase19Data(final=True)
    if name in {"raw", "age", "talon"}:
        controller: Any = RawController(data, name, deferred=True)
    else:
        if checkpoint is None:
            raise ValueError(f"checkpoint required for {name}")
        controller = ModelController(data, checkpoint, device, deferred=True)
    by_cat: dict[int, list[int]] = defaultdict(list)
    rows_out: list[dict[str, Any]] = []
    for key, indices in sorted(groups(data).items()):
        if isinstance(controller, RawController):
            controller.reset()
            decisions = controller.process_track(key, phase="known_audit")
        else:
            decisions, _ = controller.process_track(key, phase="known_audit")
        by_row = {d["row_key"]: d for d in decisions}
        local_known: int | None = None
        for i in indices:
            row = data.rows[i]
            d = by_row[row["row_key"]]
            if d["action"] == "KNOWN" and d.get("semantic_id") is not None:
                local_known = int(d["semantic_id"])
            pred = -1 if local_known is None else local_known
            gt = int(row["gt_category_id_common"])
            ok = int(pred == gt)
            by_cat[gt].append(ok)
            rows_out.append({"row_key": row["row_key"], "track_key": key,
                             "gt_category_evaluator_only": gt, "action": d["action"],
                             "predicted_known_id": pred, "correct": bool(ok),
                             "readiness": float(d.get("readiness", 0.0))})
    values = [v for vals in by_cat.values() for v in vals]
    result = {
        "protocol": "trackocd_iclr27_phase19_public_known_audit_after_freeze",
        "candidate": name,
        "freeze_marker": str((OUT / "completion/public_predictions.frozen").resolve()),
        "rows": len(values),
        "tracks": len(groups(data)),
        "categories": len(by_cat),
        "micro_accuracy": float(np.mean(values)) if values else 0.0,
        "category_macro_accuracy": float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0,
        "by_category": {str(c): {"correct": int(sum(v)), "rows": len(v),
                                  "accuracy": float(np.mean(v))} for c, v in sorted(by_cat.items())},
        "records": rows_out,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    marker = OUT / "completion/public_predictions.frozen"
    if not marker.exists():
        raise SystemExit("prediction freeze marker is required before known-label measurement")
    device = torch.device(args.device)
    candidates = {
        "raw": None,
        "age": None,
        "talon": None,
        "main": OUT / "checkpoints/main_fold0_best.pt",
        "fallback_a": OUT / "checkpoints/final_fallback_a_best.pt",
    }
    result = {"protocol": "trackocd_iclr27_phase19_public_known_audit_after_freeze",
              "candidates": {name: run_candidate(name, ckpt, device)
                             for name, ckpt in candidates.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(args.out)
    print(json.dumps({k: {m: v[m] for m in ("rows", "tracks", "categories", "micro_accuracy", "category_macro_accuracy")}
                      for k, v in result["candidates"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
