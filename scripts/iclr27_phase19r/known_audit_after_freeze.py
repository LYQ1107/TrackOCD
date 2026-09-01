"""Measure supported-known safety after the public prediction freeze."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.models.controller import RCMSOCD
from src.iclr27_phase19r.models.known_osr import GaussianController, RawPersistentController, TALONStyleController
from src.iclr27_phase19r.runtime.runner import ModelStreamController


ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/iclr27_phase19r"


def groups(data: Phase19RData) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(data.rows):
        if row.get("role17") == "known_audit" and row.get("gt_role_common") == "supported_known":
            out[f"v{int(row['video_id'])}:p{int(row['track_id'])}"].append(i)
    for k in out: out[k].sort(key=lambda i: (int(data.rows[i]["event_rank"]), i))
    return dict(out)


def make_model(data: Phase19RData, checkpoint: Path, device: torch.device) -> ModelStreamController:
    ck = torch.load(checkpoint, map_location="cpu")
    model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=16, known_bias=torch.from_numpy(data.known_bias))
    model.load_state_dict(ck["model_state"]); model.to(device).eval()
    return ModelStreamController(model, max_states=16, allow_defer=True,
                                 tau_ready=model.tau_ready, tau_known=model.tau_known, tau_assign=model.tau_assign)


def run_candidate(name: str, data: Phase19RData, checkpoint: Path | None, device: torch.device) -> dict[str, Any]:
    if name == "raw": controller: Any = RawPersistentController(data, deferred=True); model_mode = False
    elif name == "age": controller = GaussianController(data, deferred=True); model_mode = False
    elif name == "talon": controller = TALONStyleController(data, deferred=True); model_mode = False
    elif name == "fallback_f_a": controller = GaussianController(data, deferred=True); model_mode = False
    else:
        if checkpoint is None: raise ValueError("checkpoint required")
        controller = make_model(data, checkpoint, device); model_mode = True
    by_cat: dict[int, list[int]] = defaultdict(list); rows_out = []
    km = torch.from_numpy(data.active_known_mask).to(device)
    for key, indices in sorted(groups(data).items()):
        if hasattr(controller, "reset_stream"): controller.reset_stream()
        if model_mode:
            decisions = []
            for pos in range(len(data.track_rows[key])):
                raw, geom, quality, _ = data.prefix(key, pos); row = data.rows[data.track_rows[key][pos]]
                got = controller.process_item(torch.from_numpy(raw).to(device), torch.from_numpy(geom).to(device), quality, int(row["video_id"]), key, km, oracle_category=None)
                decisions.append({"row_key": row["row_key"], "action": got["action"], "semantic_id": got.get("semantic_id"), "known_index": got.get("known_index"), "quality": quality})
        else:
            decisions = controller.process_track(key, phase="known_audit", eval_category=None)
        by_row = {d["row_key"]: d for d in decisions}; local_known: int | None = None
        for i in indices:
            row = data.rows[i]; d = by_row[row["row_key"]]
            if d.get("action") == "KNOWN":
                if d.get("known_index") is not None:
                    local_known = int(data.supported_ids[int(d["known_index"])])
                elif d.get("semantic_id") is not None:
                    local_known = int(d["semantic_id"])
            pred = -1 if local_known is None else local_known; gt = int(row["gt_category_id_common"]); ok = int(pred == gt); by_cat[gt].append(ok)
            rows_out.append({"row_key": row["row_key"], "track_key": key, "gt_category_evaluator_only": gt,
                             "action": d.get("action"), "predicted_known_id": pred, "correct": bool(ok),
                             "readiness": float(d.get("quality", d.get("readiness", 0.0)))})
    vals = [x for v in by_cat.values() for x in v]
    return {"protocol": "trackocd_iclr27_phase19r_known_audit_after_freeze", "candidate": name,
            "rows": len(vals), "tracks": len(groups(data)), "categories": len(by_cat),
            "micro_accuracy": float(np.mean(vals)) if vals else 0.0,
            "category_macro_accuracy": float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0,
            "by_category": {str(c): {"correct": int(sum(v)), "rows": len(v), "accuracy": float(np.mean(v))} for c, v in sorted(by_cat.items())},
            "records": rows_out}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--device", default="cpu"); p.add_argument("--final-checkpoint", type=Path, required=True); a = p.parse_args()
    marker = OUT / "completion/public_predictions.frozen"
    if not marker.exists(): raise SystemExit("prediction freeze marker is required")
    data = Phase19RData(final=True); dev = torch.device(a.device)
    candidates = {"raw": None, "age": None, "talon": None, "main": a.final_checkpoint, "fallback_f_a": None}
    result = {"protocol": "trackocd_iclr27_phase19r_known_audit_after_freeze", "freeze_marker": str(marker.resolve()),
              "candidates": {k: run_candidate(k, data, v, dev) for k, v in candidates.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_name(a.out.name + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, a.out)
    print(json.dumps({k: {m: v[m] for m in ("rows", "tracks", "categories", "micro_accuracy", "category_macro_accuracy")} for k, v in result["candidates"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
