"""Evaluate repaired candidates on fixed held-known L0/L1/L2 streams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate, load_events


def ladder_events(data: Phase19RData, ladder: str) -> list[dict[str, Any]]:
    events = load_events(data.fold)
    if ladder == "L2": return events
    out = []
    for e in events:
        keys = list(e["source_tracklet_keys"]) + [e["target_tracklet_key"]]
        qualities = []
        reliable = True
        for k in keys:
            for pos in range(len(data.track_rows[k])):
                _, _, q, _ = data.prefix(k, pos); qualities.append(q)
                row = data.rows[data.track_rows[k][pos]]
                if ladder == "L0" and not (row.get("assigned") == "1" and float(row.get("row_iou", 0.0)) >= .5): reliable = False
        if ladder == "L0" and reliable: out.append(e)
        elif ladder == "L1" and qualities and float(sum(qualities) / len(qualities)) >= .35: out.append(e)
    return out


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--device", default="cpu"); a = p.parse_args(); dev = torch.device(a.device)
    result: dict[str, Any] = {"protocol": "trackocd_iclr27_phase19r_internal_ladder_evaluation", "ladders": {}}
    for ladder in ("L0", "L1", "L2"):
        result["ladders"][ladder] = {}
        for name in ("raw", "age", "talon", "main", "fallback"):
            rows = []
            for fold in range(4):
                data = Phase19RData(fold); events = ladder_events(data, ladder)
                if name in {"raw", "age", "talon", "fallback"}:
                    cand = "fallback" if name == "fallback" else name
                    rows.append(evaluate_candidate(cand, data, None, dev, events))
                else:
                    rows.append(evaluate_candidate("main", data, Path(f"outputs/iclr27_phase19r/checkpoints/fold{fold}_best_internal.pt"), dev, events))
            result["ladders"][ladder][name] = {"event_count_by_fold": [int(x["events"]) for x in rows], "folds": rows,
                                                 "category_macro_mean": float(sum(x["metrics"]["category_macro_reuse"] for x in rows) / 4.0),
                                                 "existing_precision_mean": float(sum(x["metrics"]["existing_precision"] for x in rows) / 4.0),
                                                 "false_merge_mean": float(sum(x["metrics"]["negative_false_merge_rate"] for x in rows) / 4.0)}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: {n: v[n]["event_count_by_fold"] for n in v} for k, v in result["ladders"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
