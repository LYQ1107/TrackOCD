#!/usr/bin/env python3
"""One-shot frozen 76-positive + 76-negative Phase88 diagnostic replay.

The held manifests are read only after TRAIN-only checkpoint freezing.  This
script never writes a checkpoint and its output is explicitly diagnostic-only:
it cannot be consumed by checkpoint/branch selection code.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("TRACKOCD_OUT", str(ROOT / "outputs/iclr27_phase88")))
SHARED = Path("/data2/usr_for_deadline/trackocd_phase88/shared_features")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iclr27_phase88.controller import CausalPersistentOCD
from src.iclr27_phase88.data_memmap import Phase88FoldData
from src.iclr27_phase88.evaluation import (
    finalize_persistent_metrics,
    normalize_event,
    replay_persistent_records,
    evaluate_known_stream_v2,
)


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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aggregate_known(per_fold: list[dict]) -> dict:
    rows = sum(int(x.get("known_rows", 0)) for x in per_fold)
    micro_sum = sum(float(x.get("known_micro", 0.0)) * int(x.get("known_rows", 0)) for x in per_fold)
    macro = [float(x.get("known_macro", 0.0)) for x in per_fold if x.get("known_categories", 0)]
    return {
        "known_micro": micro_sum / max(rows, 1),
        "known_macro": sum(macro) / max(len(macro), 1),
        "known_rows": rows,
        "known_categories": sum(int(x.get("known_categories", 0)) for x in per_fold),
        "per_fold": per_fold,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=str(OUT / "audit/c0v2_train_selection.json"))
    ap.add_argument("--tag", default="frozen_c0v2_76plus76_diagnostic")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--memmap-root", default=str(SHARED))
    ap.add_argument("--support-mode", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--architecture", choices=("baseline", "h3"), default="baseline")
    args = ap.parse_args()
    selection_path = Path(args.selection).resolve()
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "FROZEN_TRAIN_ONLY":
        raise RuntimeError("selection must be frozen from TRAIN validation before held diagnostic")
    pos_path = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
    neg_path = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
    pos = [normalize_event(x) for x in read_jsonl(pos_path)]
    neg = [normalize_event(x) for x in read_jsonl(neg_path)]
    if len(pos) != 76 or len(neg) != 76:
        raise RuntimeError(f"registered diagnostic denominator changed: pos={len(pos)} neg={len(neg)}")
    by_fold: dict[int, list[dict]] = {f: [] for f in range(4)}
    for event in pos + neg:
        by_fold[int(event["fold"])].append(event)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    folds: list[dict] = []
    all_records: list[dict] = []
    known_by_fold: list[dict] = []
    selected_rows = selection.get("selected_fold_checkpoints", selection.get("folds"))
    if not selected_rows or len(selected_rows) != 4:
        raise RuntimeError("selection must contain four frozen fold checkpoints")
    fold_modes = {
        bool(row.get("support_mode", row.get("metrics", {}).get("support_mode", selection.get("support_mode", False))))
        for row in selected_rows
    }
    if len(fold_modes) != 1:
        raise RuntimeError("INCONSISTENT_FROZEN_SUPPORT_MODE")
    frozen_support_mode = next(iter(fold_modes))
    if args.support_mode == "on":
        requested_support_mode = True
    elif args.support_mode == "off":
        requested_support_mode = False
    else:
        requested_support_mode = frozen_support_mode
    if selection.get("candidate") == "H2_EQUAL_BUDGET" and requested_support_mode:
        raise RuntimeError("H2_EQUAL_SUPPORT_MODE_MUST_BE_FALSE")
    for row in selected_rows:
        fold = int(row["fold"])
        ckpt = Path(row["checkpoint"]).resolve()
        if sha(ckpt) != row["checkpoint_sha256"]:
            raise RuntimeError(f"frozen checkpoint hash changed for fold {fold}")
        data = Phase88FoldData(fold, args.memmap_root)
        payload = torch.load(ckpt, map_location=device)
        if args.architecture == "h3":
            from src.iclr27_phase89.hierarchical import HierarchicalPersistentOCD
            model = HierarchicalPersistentOCD(
                torch.from_numpy(__import__("numpy").asarray(data.known_prototypes)),
                torch.from_numpy(__import__("numpy").asarray(data.active_known_mask)),
                max_states=16,
            ).to(device)
        else:
            model = CausalPersistentOCD(
                torch.from_numpy(__import__("numpy").asarray(data.known_prototypes)),
                torch.from_numpy(__import__("numpy").asarray(data.active_known_mask)),
                max_states=16,
            ).to(device)
        model.load_state_dict(payload["model"], strict=False)
        model.eval()
        events = by_fold[fold]
        records, diagnostic = replay_persistent_records(model, data, events, device, support_mode=requested_support_mode)
        known = evaluate_known_stream_v2(model, data, device)
        metrics = finalize_persistent_metrics(records, known)
        all_records.extend(records)
        known_by_fold.append(known)
        folds.append({
            "fold": fold,
            "checkpoint": str(ckpt),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "train_selection_score": row["selection_score"],
            "events": len(events),
            "positive_events": sum(e.get("polarity") == "positive" for e in events),
            "negative_events": sum(e.get("polarity") != "positive" for e in events),
            "metrics": metrics,
            "known_metrics": known,
            "diagnostic": diagnostic,
        })
    aggregate_known_metrics = aggregate_known(known_by_fold)
    aggregate_metrics = finalize_persistent_metrics(all_records, aggregate_known_metrics)
    out = {
        "schema_version": "trackocd.phase88.frozen_76plus76_diagnostic.v2",
        "phase": 88,
        "tag": args.tag,
        "architecture": args.architecture,
        "status": "DIAGNOSTIC_ONLY_DO_NOT_SELECT",
        "selection_source": str(selection_path),
        "selection_sha256": sha(selection_path),
        "diagnostic_denominator": {"positive": 76, "negative": 76, "total": 152},
        "held_manifest_hashes": {"positive": sha(pos_path), "negative": sha(neg_path)},
        "folds": folds,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_known_metrics": aggregate_known_metrics,
        "records": all_records,
        "support_mode": requested_support_mode,
        "support_mode_source": "frozen_selection" if args.support_mode == "auto" else f"cli_{args.support_mode}",
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
        "used_for_selection": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    root = OUT / "diagnostic" / args.tag
    atomic_json(root / "metrics.json", out)
    atomic_json(root / "done.json", {"status": "DONE", "diagnostic_only": True, "metrics": str((root / "metrics.json").resolve())})
    print(json.dumps({"status": out["status"], "aggregate_metrics": aggregate_metrics, "folds": folds}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
