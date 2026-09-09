#!/usr/bin/env python3
"""Auxiliary deterministic stream-level OLD/NEW discovery evaluation."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("TRACKOCD_OUT", str(ROOT / "outputs/iclr27_phase88")))
MEMMAP = os.environ.get("TRACKOCD_MEMMAP", "/data2/usr_for_deadline/trackocd_phase88/shared_features")
sys.path.insert(0, str(ROOT))
from src.iclr27_phase88.controller import CausalPersistentOCD
from src.iclr27_phase88.data import FeatureStore
from src.iclr27_phase88.data_memmap import Phase88FoldData
from src.iclr27_phase88.runtime import CausalPersistentRuntime
from src.iclr27_phase88.standard_discovery_metrics import stream_discovery_metrics


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--selection", default=str(OUT / "audit/h2_equal_frozen_selection_corrected.json"))
    ap.add_argument("--support-mode", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--architecture", choices=["baseline", "h3"], default="baseline")
    ap.add_argument("--manifest", default=str(OUT / "manifests/standard_gcd_stream_v1.json"))
    ap.add_argument("--output", default=str(OUT / "audit/standard_gcd_stream_metrics_corrected.json"))
    ap.add_argument("--status", default="PSEUDO_NOVEL_TRAIN_REPORTING_ONLY")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    selection_path = Path(args.selection).resolve()
    selection = json.loads(selection_path.read_text())
    selected_rows = selection.get("selected_fold_checkpoints", selection.get("folds"))
    if not selected_rows or len(selected_rows) != 4:
        raise RuntimeError("selection must contain four frozen fold checkpoints")
    fold_modes = {bool(r.get("support_mode", r.get("metrics", {}).get("support_mode", selection.get("support_mode", False)))) for r in selected_rows}
    if len(fold_modes) != 1:
        raise RuntimeError("INCONSISTENT_FROZEN_SUPPORT_MODE")
    frozen_support_mode = next(iter(fold_modes))
    if args.support_mode == "on":
        support_mode = True
    elif args.support_mode == "off":
        support_mode = False
    else:
        support_mode = frozen_support_mode
    if selection.get("candidate") == "H2_EQUAL_BUDGET" and support_mode:
        raise RuntimeError("H2_EQUAL_SUPPORT_MODE_MUST_BE_FALSE")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    fold_outputs = []
    for fmeta in manifest["folds"]:
        fold = int(fmeta["fold"])
        row = next((r for r in selected_rows if int(r.get("fold", -1)) == fold), None)
        if row is None or not row.get("checkpoint"):
            raise RuntimeError(f"MISSING_FROZEN_CHECKPOINT_ROW fold={fold}")
        ckpt = Path(str(row["checkpoint"])).resolve()
        data = Phase88FoldData(fold, MEMMAP)
        if args.architecture == "h3":
            from src.iclr27_phase89.hierarchical import HierarchicalPersistentOCD
            model = HierarchicalPersistentOCD(torch.from_numpy(np.asarray(data.known_prototypes)).clone(), torch.from_numpy(np.asarray(data.active_known_mask)).clone(), max_states=16).to(device)
        else:
            model = CausalPersistentOCD(torch.from_numpy(np.asarray(data.known_prototypes)).clone(), torch.from_numpy(np.asarray(data.active_known_mask)).clone(), max_states=16).to(device)
        payload = torch.load(ckpt, map_location=device); model.load_state_dict(payload["model"], strict=False); model.eval()
        runtime = CausalPersistentRuntime(model, FeatureStore(fold, data), device); runtime.reset_stream()
        km = torch.from_numpy(np.asarray(data.active_known_mask, dtype=bool)).to(device)
        rows = []
        for item in fmeta["tracks"]:
            key = str(item["track_key"]); video = int(item["video_id"])
            result = runtime.process_track(key, video, km, oracle_category_for_eval=None, support_mode=support_mode)
            action = result.get("final_action")
            session = result.get("session")
            if action == "KNOWN":
                token = f"K:{int(session.committed_known_index) if session and session.committed_known_index is not None else -1}"
            elif action in {"EXISTING", "NEW"}:
                sid = None
                if session is not None and session.committed_global_index is not None and session.committed_global_index < len(runtime.memory.states):
                    sid = runtime.memory.states[session.committed_global_index].sid
                if sid is None and action == "NEW" and runtime.memory.states:
                    sid = runtime.memory.states[-1].sid
                token = f"A:{int(sid)}" if sid is not None else f"U:{key}"
            else:
                token = f"U:{key}"
            category = int(item.get("category_eval", data.category(key) if hasattr(data, "category") else data.track_category[key]))
            row = {"fold": fold, "stream_id": f"fold{fold}", "track_key": key,
                   "predicted_token": f"fold{fold}|{token}", "target_category": str(category),
                   "split": "old" if item["role"] in {"known_anchor", "old"} else "new", "action": action}
            rows.append(row)
        metric = stream_discovery_metrics(rows)
        fold_outputs.append({"fold": fold, "track_count": len(rows), "known_anchor_count": fmeta.get("known_anchor_count", 0), "metrics": metric, "rows": rows, "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": sha(ckpt)})
    def aggregate_fold_metrics(items):
        total_rows = sum(int(x["metrics"].get("rows", 0)) for x in items)
        old_rows = sum(int(x["metrics"].get("old_rows", 0)) for x in items)
        new_rows = sum(int(x["metrics"].get("new_rows", 0)) for x in items)
        all_correct = sum(int(x["metrics"].get("all_correct", 0)) for x in items)
        old_correct = sum(int(x["metrics"].get("old_correct", 0)) for x in items)
        new_correct = sum(int(x["metrics"].get("new_correct", 0)) for x in items)
        old_acc = old_correct / max(old_rows, 1); new_acc = new_correct / max(new_rows, 1)
        return {
            "rows": total_rows, "old_rows": old_rows, "new_rows": new_rows,
            "all_correct": all_correct, "old_correct": old_correct, "new_correct": new_correct,
            "all_acc_micro": all_correct / max(total_rows, 1),
            "old_acc_micro": old_acc, "pseudo_novel_acc_micro": new_acc,
            "h_score_micro": 2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-12),
            "all_acc_macro": float(np.mean([x["metrics"].get("all_acc", 0.0) for x in items])),
            "old_acc_macro": float(np.mean([x["metrics"].get("old_acc", 0.0) for x in items])),
            "pseudo_novel_acc_macro": float(np.mean([x["metrics"].get("pseudo_novel_acc", x["metrics"].get("new_acc", 0.0)) for x in items])),
            "h_score_macro": float(np.mean([x["metrics"].get("h_score", 0.0) for x in items])),
            "nmi_macro": float(np.mean([x["metrics"].get("nmi", 0.0) for x in items])),
            "ari_macro": float(np.mean([x["metrics"].get("ari", 0.0) for x in items])),
            "nmi_weighted": float(sum(x["metrics"].get("nmi", 0.0) * x["metrics"].get("rows", 0) for x in items) / max(total_rows, 1)),
            "ari_weighted": float(sum(x["metrics"].get("ari", 0.0) * x["metrics"].get("rows", 0) for x in items) / max(total_rows, 1)),
        }
    aggregate = aggregate_fold_metrics(fold_outputs)
    out = {"schema_version": "trackocd.phase88.standard_gcd_stream_metrics.v2", "phase": 88,
           "status": args.status, "protocol": "standard_gcd_stream_v2_fold_local", "folds": fold_outputs,
           "aggregate": aggregate, "manifest": str(manifest_path.resolve()), "manifest_sha256": sha(manifest_path),
           "selection": str(selection_path), "selection_sha256": sha(selection_path),
           "support_mode": support_mode, "support_mode_source": "frozen_selection" if args.support_mode == "auto" else f"cli_{args.support_mode}",
           "architecture": args.architecture,
           "future_rows_or_tracks": False, "category_text_used_for_order": False, "held_tuning": False,
           "public_dev_q1_sealed_accessed": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic_json(Path(args.output), out)
    print(json.dumps({"status": out["status"], "aggregate": aggregate, "folds": [{"fold": x["fold"], "metrics": x["metrics"]} for x in fold_outputs]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
