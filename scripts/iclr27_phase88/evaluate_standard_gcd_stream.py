#!/usr/bin/env python3
"""Auxiliary deterministic stream-level OLD/NEW discovery evaluation."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"
MEMMAP = "/data2/usr_for_deadline/trackocd_phase88/shared_features"
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
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    if device.type == "cuda": torch.cuda.set_device(device)
    manifest_path = OUT / "manifests/standard_gcd_stream_v1.json"
    manifest = json.loads(manifest_path.read_text())
    fold_outputs = []
    all_rows = []
    for fmeta in manifest["folds"]:
        fold = int(fmeta["fold"])
        ckpt = OUT / "checkpoints" / f"h2_equal_f{fold}.pt"
        data = Phase88FoldData(fold, MEMMAP)
        model = CausalPersistentOCD(torch.from_numpy(np.asarray(data.known_prototypes)).clone(), torch.from_numpy(np.asarray(data.active_known_mask)).clone(), max_states=16).to(device)
        payload = torch.load(ckpt, map_location=device); model.load_state_dict(payload["model"], strict=False); model.eval()
        runtime = CausalPersistentRuntime(model, FeatureStore(fold, data), device); runtime.reset_stream()
        km = torch.from_numpy(np.asarray(data.active_known_mask, dtype=bool)).to(device)
        rows = []
        for item in fmeta["tracks"]:
            key = str(item["track_key"]); video = int(item["video_id"])
            result = runtime.process_track(key, video, km, oracle_category_for_eval=None, support_mode=True)
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
            category = int(data.category(key) if hasattr(data, "category") else data.track_category[key])
            row = {"fold": fold, "stream_id": f"fold{fold}", "track_key": key,
                   "predicted_token": f"fold{fold}|{token}", "target_category": str(category),
                   "split": "old" if item["role"] == "known_anchor" else "new", "action": action}
            rows.append(row); all_rows.append(row)
        metric = stream_discovery_metrics(rows)
        fold_outputs.append({"fold": fold, "track_count": len(rows), "known_anchor_count": fmeta.get("known_anchor_count", 0), "metrics": metric, "rows": rows, "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": sha(ckpt)})
    aggregate = stream_discovery_metrics(all_rows)
    out = {"schema_version": "trackocd.phase88.standard_gcd_stream_metrics.v1", "phase": 88,
           "status": "AUXILIARY_REPORTING_ONLY", "protocol": "standard_gcd_stream_v1", "folds": fold_outputs,
           "aggregate": aggregate, "manifest": str(manifest_path.resolve()), "manifest_sha256": sha(manifest_path),
           "future_rows_or_tracks": False, "category_text_used_for_order": False, "held_tuning": False,
           "public_dev_q1_sealed_accessed": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic_json(OUT / "audit/standard_gcd_stream_metrics.json", out)
    print(json.dumps({"status": out["status"], "aggregate": aggregate, "folds": [{"fold": x["fold"], "metrics": x["metrics"]} for x in fold_outputs]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
