"""Known-class micro/macro scores for the standard held-known lane."""
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


def one(fold: int, candidate: str, checkpoint: Path | None, device: torch.device) -> dict[str, Any]:
    data = Phase19Data(fold)
    controller: Any
    if candidate in {"raw", "age", "talon"}:
        controller = RawController(data, candidate, deferred=False)
    else:
        controller = ModelController(data, checkpoint, device, deferred=False)
    val_videos = set(int(v) for v in data.fold_record["validation_videos"])
    visible = data.supported_set - data.held_categories
    keys = [k for k in data.track_rows if data.track_cat_eval[k] in visible and data.track_video[k] in val_videos]
    by_cat: dict[int, list[int]] = defaultdict(list)
    for key in sorted(keys):
        if isinstance(controller, RawController):
            controller.reset(); decisions = controller.process_track(key, phase="internal_known")
        else:
            decisions, _ = controller.process_track(key, phase="internal_known")
        local_known: int | None = None
        lookup = {d["row_key"]: d for d in decisions}
        for i in data.track_rows[key]:
            row = data.rows[i]; d = lookup[row["row_key"]]
            if d["action"] == "KNOWN" and d.get("semantic_id") is not None:
                local_known = int(d["semantic_id"])
            pred = -1 if local_known is None else local_known
            by_cat[data.track_cat_eval[key]].append(int(pred == data.track_cat_eval[key]))
    vals = [x for v in by_cat.values() for x in v]
    return {"fold": fold, "candidate": candidate, "rows": len(vals), "tracks": len(keys),
            "categories": len(by_cat), "micro_accuracy": float(np.mean(vals)) if vals else 0.0,
            "category_macro_accuracy": float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0,
            "by_category": {str(c): {"correct": int(sum(v)), "rows": len(v), "accuracy": float(np.mean(v))}
                            for c, v in sorted(by_cat.items())}}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--device", default="cpu")
    a = p.parse_args(); device = torch.device(a.device)
    root = Path(__file__).resolve().parents[2]; ckpt = root / "outputs/iclr27_phase19/checkpoints"
    spec = {"raw": None, "age": None, "talon": None,
            "main": ckpt / "main_fold{fold}_best.pt", "fallback_a": ckpt / "fallback_a_fold{fold}_best.pt"}
    rows = []
    for candidate, template in spec.items():
        for fold in range(4):
            path = None if template is None else Path(str(template).format(fold=fold))
            rows.append(one(fold, candidate, path, device))
    result = {"protocol": "trackocd_iclr27_phase19_internal_known_micro_macro", "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); tmp.replace(a.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
