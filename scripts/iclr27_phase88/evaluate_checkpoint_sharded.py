#!/usr/bin/env python3
"""Causal event validation with resumable atomic record shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("TRACKOCD_OUT", str(ROOT / "outputs/iclr27_phase88")))

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
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--event-tag", default="fix2")
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--memmap-root", default="/data2/usr_for_deadline/trackocd_phase88/shared_features")
    ap.add_argument("--support-mode", action="store_true")
    ap.add_argument("--architecture", choices=("baseline", "h3"), default="baseline")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = Phase88FoldData(args.fold, args.memmap_root)
    ckpt = Path(args.checkpoint).resolve()
    payload = torch.load(ckpt, map_location=device)
    if args.architecture == "h3":
        from src.iclr27_phase89.hierarchical import HierarchicalPersistentOCD
        model = HierarchicalPersistentOCD(torch.from_numpy(__import__('numpy').asarray(data.known_prototypes)), torch.from_numpy(__import__('numpy').asarray(data.active_known_mask)), max_states=16).to(device)
    else:
        model = CausalPersistentOCD(torch.from_numpy(__import__('numpy').asarray(data.known_prototypes)), torch.from_numpy(__import__('numpy').asarray(data.active_known_mask)), max_states=16).to(device)
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    events = [normalize_event(e) for e in read_jsonl(OUT / "manifests" / f"val_events_v2_{args.event_tag}_f{args.fold}.jsonl")]
    if args.max_events > 0:
        events = events[:args.max_events]
    root = OUT / "validation" / args.tag
    root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    shard_diags: list[dict] = []
    for shard_index, start in enumerate(range(0, len(events), int(args.shard_size))):
        shard_events = events[start:start + int(args.shard_size)]
        rec_path = root / f"records_shard_{shard_index:04d}.json"
        done_path = root / f"records_shard_{shard_index:04d}.done"
        if rec_path.exists() and done_path.exists():
            payload_shard = json.loads(rec_path.read_text())
            all_records.extend(payload_shard["records"])
            shard_diags.append(payload_shard.get("diagnostic", {}))
            continue
        records, diagnostic = replay_persistent_records(model, data, shard_events, device, support_mode=args.support_mode)
        atomic_json(rec_path, {"shard": shard_index, "start": start, "events": len(shard_events), "records": records, "diagnostic": diagnostic})
        atomic_json(done_path, {"status": "DONE", "records": str(rec_path.resolve())})
        all_records.extend(records)
        shard_diags.append(diagnostic)
    known_path = root / "known_metrics.json"
    if known_path.exists():
        known = json.loads(known_path.read_text())
    else:
        known = evaluate_known_stream_v2(model, data, device)
        atomic_json(known_path, known)
    metrics = finalize_persistent_metrics(all_records, known)
    final = {
        "schema_version": "trackocd.phase88.sharded_validation.v1",
        "phase": 88, "tag": args.tag, "fold": args.fold, "architecture": args.architecture,
        "events": len(events), "shard_size": int(args.shard_size),
        "checkpoint": str(ckpt), "checkpoint_sha256": sha(ckpt),
        "metrics": metrics, "known_metrics": known,
        "diagnostic": {"shards": len(shard_diags), "records": len(all_records), "shard_diagnostics": shard_diags},
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
    }
    atomic_json(root / "final_metrics.json", final)
    atomic_json(root / "final.done", {"status": "DONE", "metrics": str((root / "final_metrics.json").resolve())})
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
