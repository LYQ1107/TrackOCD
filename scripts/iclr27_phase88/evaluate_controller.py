#!/usr/bin/env python3
"""Frozen TRAIN validation and diagnostic event replay for Phase87 C0."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict

import torch

torch.set_num_threads(2)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase88"
CHECKPOINT_ROOT = pathlib.Path("/data2/usr_for_deadline/trackocd_phase88/project_outputs/checkpoints")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iclr27_phase88.controller import CausalPersistentOCD
from src.iclr27_phase88.data import FeatureStore, reliability_prefix
from src.iclr27_phase88.rollout import rollout_event


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def events_from(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def normalize_event(event: dict) -> dict:
    """Bridge the frozen Phase19R held-manifest names without changing labels."""
    normalized = dict(event)
    if "target_track_key" not in normalized:
        normalized["target_track_key"] = normalized["target_tracklet_key"]
    if "source_track_keys" not in normalized:
        normalized["source_track_keys"] = list(normalized["source_tracklet_keys"])
    if "polarity" not in normalized:
        normalized["polarity"] = "positive" if normalized.get("kind") == "positive_existing" else "negative"
    return normalized


def atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def replay(model: CausalPersistentOCD, store: FeatureStore, events: list[dict], device: torch.device, retain_records: bool, support_mode: bool = False) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    action_counts: Counter[str] = Counter()
    positive = negative = correct = premature = unresolved = false_merge = false_commit = 0
    existing_predictions = positive_existing = 0
    categories: defaultdict[str, int] = defaultdict(int)
    videos: defaultdict[str, int] = defaultdict(int)
    losses: list[float] = []
    for event in events:
        # The prefix is computed from row-side reliability only for scoring;
        # it is never passed to the model.
        if "reliable_prefix_for_loss_only" in event:
            reliable = int(event["reliable_prefix_for_loss_only"])
        else:
            reliable = reliability_prefix(store.table, event["target_track_key"], max_len=16)
        reliable = max(1, min(16, reliable))
        with torch.no_grad():
            loss, trace_info = rollout_event(model, store, event, train=False, teacher_probability=0.0, support_mode=support_mode)
        losses.append(float(loss.detach().cpu()))
        trace = trace_info["trace"]
        for row in trace:
            action_counts[row["predicted"]] += 1
        is_positive = event.get("polarity") == "positive" or event.get("kind") == "positive_existing"
        positive += int(is_positive)
        negative += int(not is_positive)
        pre = trace[: max(0, reliable - 1)]
        post = trace[max(0, reliable - 1) :]
        pre_commits = [r for r in pre if r["predicted"] in {"EXISTING", "KNOWN", "NEW"}]
        post_commits = [r for r in post if r["predicted"] in {"EXISTING", "KNOWN", "NEW"}]
        first = post_commits[0] if post_commits else None
        premature += int(bool(pre_commits))
        unresolved += int(first is None)
        if first is not None:
            action = first["predicted"]
            if is_positive:
                if action in {"EXISTING", "KNOWN"} and not pre_commits:
                    correct += 1
                    positive_existing += 1
                    categories[str(event.get("target_category_for_loss_only", event.get("category_gt_denominator_only", "unknown")))] += 1
                    videos[str(event.get("target_video", "unknown"))] += 1
                false_commit += int(action == "NEW")
            else:
                false_commit += 1
                false_merge += int(action in {"EXISTING", "KNOWN"})
            existing_predictions += int(action in {"EXISTING", "KNOWN"})
        if retain_records:
            records.append({
                "event_id": event.get("event_id", event.get("event_key")),
                "fold": event.get("fold"),
                "polarity": "positive" if is_positive else "negative",
                "reliable_prefix_scored": reliable,
                "first_post_prefix_action": first["predicted"] if first else None,
                "first_post_prefix_position": first["position"] if first else None,
                "premature": bool(pre_commits),
                "unresolved": first is None,
                "correct_commit_ct": bool(is_positive and first is not None and first["predicted"] in {"EXISTING", "KNOWN"} and not pre_commits),
                "negative_false_merge": bool((not is_positive) and first is not None and first["predicted"] in {"EXISTING", "KNOWN"}),
                "trace": trace,
            })
    total = max(1, len(events))
    pos_total = max(1, positive)
    neg_total = max(1, negative)
    result = {
        "events": len(events),
        "positive_events": positive,
        "negative_events": negative,
        "commit_ct_correct": correct,
        "commit_ct_eligible": positive,
        "commit_ct_recall": correct / pos_total,
        "category_coverage": len(categories),
        "video_coverage": len(videos),
        "existing_precision": positive_existing / max(1, existing_predictions),
        "existing_recall": correct / pos_total,
        "negative_false_merge_rate": false_merge / neg_total,
        "negative_false_commit_rate": false_commit / neg_total,
        "premature_rate": premature / total,
        "unresolved_rate": unresolved / total,
        "action_counts": dict(action_counts),
        "mean_loss": sum(losses) / max(1, len(losses)),
        "by_category_correct": dict(sorted(categories.items())),
        "by_video_correct": dict(sorted(videos.items())),
    }
    return result, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["val", "held"], default="val")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--retain-records", action="store_true")
    ap.add_argument("--max-events", type=int, default=0, help="bounded contract smoke limit; 0 means all")
    ap.add_argument("--support-mode", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = pathlib.Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location=device)
    model = CausalPersistentOCD(max_states=16).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    store = FeatureStore()
    if args.split == "val":
        event_path = OUT / "manifests" / f"val_events_f{args.fold}.jsonl"
    else:
        event_path = ROOT / "outputs" / "iclr27_phase19r" / "manifests" / "held_known_positive_events.jsonl"
        neg_path = ROOT / "outputs" / "iclr27_phase19r" / "manifests" / "held_known_negative_events.jsonl"
    if args.split == "val":
        events = events_from(event_path)
    else:
        events = [normalize_event(e) for e in events_from(event_path) if int(e.get("fold", -1)) == args.fold]
        events += [normalize_event(e) for e in events_from(neg_path) if int(e.get("fold", -1)) == args.fold]
    if args.max_events > 0:
        events = events[: args.max_events]
    metrics, records = replay(model, store, events, device, args.retain_records, support_mode=args.support_mode)
    result = {"schema_version": "trackocd.phase87.controller_replay.v1", "phase": 87, "tag": args.tag, "fold": args.fold, "split": args.split, "support_mode": args.support_mode, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint), "metrics": metrics, "records_path": None, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_or_text_as_model_input": False, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    out = OUT / "metrics" / f"{args.tag}.json"
    if args.retain_records:
        rp = OUT / "metrics" / f"{args.tag}_records.json"
        atomic_json(rp, {"records": records})
        result["records_path"] = str(rp.resolve())
    atomic_json(out, result)
    done = OUT / "completion" / f"{args.tag}.done"
    atomic_json(done, {"status": "DONE", "metrics": str(out.resolve())})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
