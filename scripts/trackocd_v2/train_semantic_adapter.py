#!/usr/bin/env python3
"""Train the single TrackOCD v2 TRAIN-only semantic adapter.

This route is intentionally downstream of the physical frontend/cache freeze.
It trains on old-role GT tracks from TAO Train, while validation is used only
for a development retrieval diagnostic.  It never loads Test labels and does
not alter the physical ``z_instance`` representation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.models.semantic_adapter import (  # noqa: E402
    SemanticAdapter,
    SemanticAdapterConfig,
    hard_negative_ranking_loss,
    multi_positive_contrastive_loss,
    prefix_consistency_loss,
)


PREFIXES = (1, 2, 4, 8, 16)
FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks"
MANIFEST_ROOT = OUTPUT_TARGET / "manifests"
SELECTION = OUTPUT_TARGET / "audit/frontend_selection.json"
REPRESENTATION = OUTPUT_TARGET / "audit/representation_decision.json"
FORMAL_MANIFEST_ROOT = OUTPUT_TARGET / "features/formal"
OUTPUT = OUTPUT_TARGET / "tables/semantic_adapter.json"
SELECTION_OUTPUT = OUTPUT_TARGET / "audit/semantic_adapter_selection.json"
CHECKPOINT = OUTPUT_TARGET / "checkpoints/semantic_adapter.pt"
LAUNCHED = OUTPUT_TARGET / "checkpoints/semantic_adapter.launched"
DONE = OUTPUT_TARGET / "checkpoints/semantic_adapter.done"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _require_final_representation() -> dict[str, Any]:
    for path, label in ((SELECTION, "frontend selection"), (REPRESENTATION, "representation decision")):
        if not path.is_file():
            raise RuntimeError(f"semantic adapter prerequisite missing: {label}: {path}")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    representation = json.loads(REPRESENTATION.read_text(encoding="utf-8"))
    if selection.get("status") != "FINAL_PHYSICAL_FRONTEND_SELECTED":
        raise RuntimeError("semantic adapter waits for FINAL_PHYSICAL_FRONTEND")
    if representation.get("status") != "FINAL_REPRESENTATION_SELECTED":
        raise RuntimeError("semantic adapter waits for FINAL_REPRESENTATION_SELECTED")
    if representation.get("formal_common_feature_cache_authorized") is not True:
        raise RuntimeError("semantic adapter waits for formal cache authorization")
    selected = str(selection.get("selected_frontend") or "")
    slug = {
        "SimOWT/Q0": "simowt",
        "OVTR-native": "ovtr",
        "COVTrack-native": "covtrack_native",
        "COVTrack-NoSemantic": "covtrack_nosem",
    }.get(selected)
    if slug is None:
        raise RuntimeError(f"unknown selected frontend: {selected}")
    cache_manifest = FORMAL_MANIFEST_ROOT / slug / "cache_manifest.json"
    if not cache_manifest.is_file() or json.loads(cache_manifest.read_text(encoding="utf-8")).get("status") != "COMPLETE":
        raise RuntimeError("semantic adapter waits for the selected frontend formal cache")
    return {
        "selected_frontend": selected,
        "selection": str(SELECTION.resolve()),
        "selection_sha256": sha256_file(SELECTION),
        "representation": str(REPRESENTATION.resolve()),
        "representation_sha256": sha256_file(REPRESENTATION),
        "formal_cache_manifest": str(cache_manifest.resolve()),
        "formal_cache_manifest_sha256": sha256_file(cache_manifest),
    }


def _load_samples(split: str, rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]], *, role: str) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        key = str(row["sample_key"])
        label = labels.get(key)
        if label is None or str(label.get("gt_split")) != role:
            continue
        payload = json.loads((FEATURE_ROOT / split / f"{key}.json").read_text(encoding="utf-8"))
        prefixes = {}
        for prefix in PREFIXES:
            vector = np.asarray(payload["prefix_features"][str(prefix)], dtype=np.float32)
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError(f"invalid {split} semantic-adapter feature for {key}, p{prefix}")
            prefixes[prefix] = vector
        samples.append({
            "sample_key": key,
            "video_id": int(row["video_id"]),
            "category_id": int(label["gt_category_id"]),
            "prefixes": prefixes,
        })
    if not samples:
        raise ValueError(f"no {role} samples in {split}")
    return samples


def _category_groups(samples: list[dict[str, Any]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[int(sample["category_id"])].append(index)
    return {category: values for category, values in groups.items() if len(values) >= 2}


def _batch(samples: list[dict[str, Any]], groups: dict[int, list[int]], rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    categories = list(groups)
    if not categories:
        raise ValueError("semantic adapter needs at least one category with two tracks")
    per_category = max(2, min(4, batch_size // max(1, min(len(categories), batch_size // 2))))
    count_categories = max(1, min(len(categories), batch_size // per_category))
    selected = rng.choice(categories, size=count_categories, replace=len(categories) < count_categories)
    prefix_features = []
    full_features = []
    labels = []
    group_ids = []
    for group_id, category in enumerate(selected):
        for index in rng.choice(groups[int(category)], size=per_category, replace=len(groups[int(category)]) < per_category):
            sample = samples[int(index)]
            prefix = int(rng.choice(PREFIXES[:-1]))
            prefix_features.append(sample["prefixes"][prefix])
            full_features.append(sample["prefixes"][16])
            labels.append(int(category))
            group_ids.append(str(sample["sample_key"]))
    # Integer group IDs are sufficient because the sampler has unique track
    # keys; same-category rows remain valid cross-track positives.
    unique_groups = {key: index for index, key in enumerate(sorted(set(group_ids)))}
    return (
        np.asarray(prefix_features, dtype=np.float32),
        np.asarray(full_features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray([unique_groups[key] for key in group_ids], dtype=np.int64),
    )


def _encode(model: Any, values: np.ndarray, torch: Any, device: Any, batch_size: int = 256) -> np.ndarray:
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            tensor = torch.from_numpy(values[start:start + batch_size]).to(device)
            result.append(model(tensor)["z_semantic"].cpu().numpy())
    return np.concatenate(result, axis=0)


def _val_retrieval(model: Any, samples: list[dict[str, Any]], torch: Any, device: Any) -> dict[str, float | int]:
    usable = [sample for sample in samples if sample["category_id"] is not None]
    if len(usable) < 2:
        return {"queries": 0, "cross_video_queries": 0, "category_recall_at_1": 0.0}
    features = np.asarray([sample["prefixes"][16] for sample in usable], dtype=np.float32)
    embeddings = _encode(model, features, torch, device)
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    hits = []
    for index, sample in enumerate(usable):
        candidates = [j for j, other in enumerate(usable) if int(other["video_id"]) != int(sample["video_id"])]
        if not candidates:
            continue
        best = candidates[int(np.argmax(similarity[index, candidates]))]
        hits.append(int(int(usable[best]["category_id"]) == int(sample["category_id"])))
    return {
        "queries": len(usable),
        "cross_video_queries": len(hits),
        "category_recall_at_1": float(np.mean(hits)) if hits else 0.0,
    }


def _atomic_torch(path: Path, payload: dict[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = ensure_output_layout()
    lineage = _require_final_representation()
    if DONE.exists() and not args.resume:
        print(json.dumps({"status": "ALREADY_DONE", "done": str(DONE.resolve())}, indent=2))
        return 0
    if LAUNCHED.exists() and not args.resume:
        try:
            marker = json.loads(LAUNCHED.read_text(encoding="utf-8"))
            pid = int(marker["pid"])
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            raise RuntimeError(f"launched marker requires explicit --resume: {LAUNCHED}")
        raise RuntimeError(f"semantic adapter is already owned by PID {pid}")

    import torch

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.resume and LAUNCHED.exists() and not CHECKPOINT.exists():
        raise RuntimeError("cannot resume semantic adapter without a valid checkpoint")
    if not LAUNCHED.exists():
        atomic_json(LAUNCHED, {"pid": os.getpid(), "started_utc": _now(), "device": str(device), "seed": args.seed, "lineage": lineage})
    train_rows = _jsonl(MANIFEST_ROOT / "tao_train_gt_tracks.jsonl")
    val_rows = _jsonl(MANIFEST_ROOT / "tao_val_gt_tracks.jsonl")
    train_labels = {str(row["sample_key"]): row for row in _jsonl(MANIFEST_ROOT / "private_tao_train_gt_track_labels.jsonl")}
    val_labels = {str(row["sample_key"]): row for row in _jsonl(MANIFEST_ROOT / "private_tao_val_gt_track_labels.jsonl")}
    train_samples = _load_samples("train", train_rows, train_labels, role="old")
    val_samples = _load_samples("val", val_rows, val_labels, role="old")
    groups = _category_groups(train_samples)
    model = SemanticAdapter(SemanticAdapterConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    start = 0
    history: list[dict[str, Any]] = []
    if args.resume and CHECKPOINT.exists():
        payload = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start = int(payload.get("step", 0))
        history = list(payload.get("history", []))
    model.to(device)
    rng = np.random.default_rng(args.seed + 7)
    steps = 2 if args.smoke else max(1, int(args.steps))
    interval = max(1, min(int(args.checkpoint_every), steps))
    for step in range(start + 1, steps + 1):
        prefix, full, labels, group_ids = _batch(train_samples, groups, rng, int(args.batch_size))
        prefix_tensor = torch.from_numpy(prefix).to(device)
        full_tensor = torch.from_numpy(full).to(device)
        label_tensor = torch.from_numpy(labels).to(device)
        group_tensor = torch.from_numpy(group_ids).to(device)
        optimizer.zero_grad(set_to_none=True)
        prefix_semantic = model(prefix_tensor)["z_semantic"]
        full_semantic = model(full_tensor)["z_semantic"]
        contrastive = multi_positive_contrastive_loss(prefix_semantic, label_tensor, group_ids=group_tensor, temperature=0.07)
        ranking = hard_negative_ranking_loss(prefix_semantic, label_tensor, group_ids=group_tensor, margin=0.10)
        consistency = prefix_consistency_loss(prefix_semantic, full_semantic)
        loss = contrastive + 0.5 * ranking + 0.2 * consistency
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite semantic-adapter loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % interval == 0 or step == steps:
            val = _val_retrieval(model, val_samples, torch, device)
            record = {"step": step, "loss": float(loss.detach().cpu()), "contrastive": float(contrastive.detach().cpu()), "ranking": float(ranking.detach().cpu()), "prefix_consistency": float(consistency.detach().cpu()), "val": val}
            history.append(record)
            _atomic_torch(CHECKPOINT, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "seed": args.seed, "history": history, "metadata": model.metadata(), "lineage": lineage}, torch)

    val = _val_retrieval(model, val_samples, torch, device)
    selection = {
        "schema_version": "trackocd.v2.semantic_adapter_selection.v1",
        "status": "SEMANTIC_ADAPTER_SELECTED",
        "generated_utc": _now(),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "selection_rule": "single registered TRAIN-only adapter; validation is diagnostic and cannot alter the physical instance representation",
        "train_tracks": len(train_samples),
        "train_categories": len(groups),
        "validation_old_tracks": len(val_samples),
        "validation_retrieval": val,
        "lineage": lineage,
        "test_semantic_accessed": False,
    }
    atomic_json(SELECTION_OUTPUT, selection)
    result = {
        "schema_version": "trackocd.v2.semantic_adapter_benchmark.v1",
        "status": "COMPLETE",
        "generated_utc": _now(),
        "steps": steps,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "history": history,
        "metadata": model.metadata(),
        "selection": str(SELECTION_OUTPUT.resolve()),
        "selection_sha256": sha256_file(SELECTION_OUTPUT),
        "test_semantic_accessed": False,
    }
    atomic_json(OUTPUT, result)
    atomic_json(DONE, {"pid": os.getpid(), "steps": steps, "checkpoint": str(CHECKPOINT.resolve()), "completed_utc": _now()})
    if LAUNCHED.exists():
        LAUNCHED.unlink()
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
