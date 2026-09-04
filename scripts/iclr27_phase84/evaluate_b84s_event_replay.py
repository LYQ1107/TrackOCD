#!/usr/bin/env python3
"""Freeze B84S and evaluate its native candidate-set decisions.

This is a post-hoc event diagnostic.  The model is loaded from the completed
TRAIN-only fold checkpoints and is never refit or calibrated here.  Native
candidate sets are reconstructed from the same Q0 stream used by the B84S
manifest; event labels/GT are used only to score the frozen decisions.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/iclr27_phase84"
BASE = Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs")
NATIVE_PATH = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
FEATURE_PATH = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
B4_PATH = Path("/data2/usr_for_deadline/trackocd_phase83/b4_native_sets/b4_native_sets_v1.npz")
MODEL_MANIFEST = BASE / "manifests/b84s_native_manifest.json"
SOURCE_CACHE = BASE / "manifests/source_track_native_vectors.npz"
SOURCE_MANIFEST = BASE / "manifests/source_track_native_cache.json"
OBS_PATH = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")
CSV_PATH = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
PREFIXES = (1, 2, 4, 8, 16)
MODEL_PREFIX = "b84s_b84s_formal_r2"
MODEL_FOLD_COUNT = 4
OUTPUT_SUFFIX = ""
RAW_ANCHOR = False
RAW_ANCHOR_BOUND = 0.05


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def box(value: Any) -> list[float] | None:
    if value is None or value == "":
        return None
    try:
        out = [float(x) for x in (value if isinstance(value, (list, tuple)) else ast.literal_eval(str(value)))]
        return out if len(out) == 4 else None
    except Exception:
        return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-8)


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def load_native() -> tuple[list[dict[str, Any]], np.ndarray, dict[tuple[int, int], list[int]], np.ndarray]:
    native = [json.loads(line) for line in NATIVE_PATH.open(encoding="utf-8") if line.strip()]
    raw = np.asarray(np.load(FEATURE_PATH, allow_pickle=False)["features"], dtype=np.float32)
    if len(native) != len(raw):
        raise RuntimeError(f"native/features mismatch: {len(native)} vs {len(raw)}")
    features = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(native):
        if box(row.get("bbox_xyxy")) is not None:
            groups[(int(row["video_id"]), int(row.get("image_id", -1)))].append(i)
    for key in groups:
        groups[key].sort(key=lambda i: (int(native[i].get("candidate_rank") or 0), int(native[i].get("physical_track_id", -1)), i))
    # The descriptor part is exactly the frozen B4 native descriptor.  It is
    # keyed back to native rows so event replay uses the runtime candidate set.
    b4 = np.load(B4_PATH, allow_pickle=False)
    desc = np.zeros((len(native), 15), dtype=np.float32)
    desc[b4["flat_indices"].astype(np.int64)] = b4["features"].astype(np.float32)
    return native, features, groups, desc


def load_gt() -> dict[str, list[float] | None]:
    return {str(r["row_key"]): box(r.get("gt_bbox_xyxy")) for r in csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))}


def load_models() -> dict[int, dict[str, np.ndarray]]:
    models: dict[int, dict[str, np.ndarray]] = {}
    for fold in range(MODEL_FOLD_COUNT):
        candidates = sorted((BASE / "checkpoints").glob(f"{MODEL_PREFIX}_f{fold}_step*.npz"))
        if not candidates:
            raise FileNotFoundError(f"no formal checkpoint for fold {fold}")
        p = candidates[-1]
        z = np.load(p, allow_pickle=False)
        models[fold] = {"w": z["w"].astype(np.float32), "b": np.asarray(z["b"], dtype=np.float32).reshape(-1)[0], "mean": z["mean"].astype(np.float32), "std": z["std"].astype(np.float32), "path": str(p.resolve()), "sha256": sha(p)}
    return models


def source_vectors() -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    z = np.load(SOURCE_CACHE, allow_pickle=False)
    keys = [str(x) for x in z["keys"].tolist()]
    return {k: i for i, k in enumerate(keys)}, z["vectors"].astype(np.float32), z["prototypes"].astype(np.float32)


def target_action(base_desc: np.ndarray, native_indices: list[int], native_features: np.ndarray, source_mean: np.ndarray, source_proto: np.ndarray, model: dict[str, np.ndarray]) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (choice, logits, 19-D features) for one native image set."""
    if not native_indices:
        return 0, np.asarray([float(model["b"])], dtype=np.float32), np.zeros((0, 19), dtype=np.float32)
    sv = norm(source_mean)
    protos = [norm(p) for p in source_proto if float(np.linalg.norm(p)) > 1e-8]
    if not protos:
        protos = [sv]
    z = native_features[np.asarray(native_indices, dtype=np.int64)]
    sims = z @ sv
    ps = np.stack([z @ p for p in protos], axis=1)
    extra = np.stack([sims, ps.max(1), ps.mean(1), ps.min(1)], axis=1).astype(np.float32)
    x = np.concatenate([base_desc, extra], axis=1).astype(np.float32)
    xn = (x - model["mean"]) / model["std"]
    logits = np.concatenate([xn @ model["w"], np.asarray([model["b"]], dtype=np.float32)])
    if RAW_ANCHOR:
        # A single fixed, bounded residual around the same-space raw anchor.
        # The frozen model's candidate logits are standardized only within
        # this causal candidate set; no event label participates in scoring.
        candidate_logits = logits[:-1]
        scale = max(float(np.std(candidate_logits)), 1e-6)
        zlogits = (candidate_logits - float(np.mean(candidate_logits))) / scale
        blended = extra[:, 0] + np.float32(RAW_ANCHOR_BOUND) * np.tanh(zlogits)
        # Explicit raw fallback for the model's DEFER action.  Empty sets
        # still return DEFER through the early branch above.
        return int(np.argmax(blended)), logits.astype(np.float32), x
    return int(np.argmax(logits)), logits.astype(np.float32), x


def evaluate() -> tuple[dict[str, Any], dict[str, Any]]:
    native, features, groups, desc = load_native()
    gt = load_gt()
    models = load_models()
    key_to_idx, source_v, source_p = source_vectors()
    obs = [json.loads(line) for line in OBS_PATH.open(encoding="utf-8") if line.strip()]
    records: list[dict[str, Any]] = []
    for event in obs:
        fold = int(event.get("fold", 0))
        source_key = str(event.get("source_tracklet_key"))
        if source_key not in key_to_idx:
            continue
        si = key_to_idx[source_key]
        # Source cache index 4 is the registered prefix-16 completed source.
        sv = source_v[4, si]
        sp = source_p[:, si]
        selected: list[dict[str, Any]] = []
        for detail in event.get("target_row_details", []):
            image_key = (int(detail.get("video_id", -1)), int(detail.get("image_id", -1)))
            inds = groups.get(image_key, [])
            model_fold = fold if fold in models else fold % MODEL_FOLD_COUNT
            choice, logits, _ = target_action(desc[np.asarray(inds, dtype=np.int64)] if inds else np.zeros((0, 15), dtype=np.float32), inds, features, sv, sp, models[model_fold])
            deferred = choice >= len(inds)
            native_index = None if deferred else int(inds[choice])
            selected_iou = 0.0 if native_index is None else iou(box(native[native_index].get("bbox_xyxy")), gt.get(str(detail.get("row_key"))))
            # Frozen same-space diagnostics for comparison, without any GT.
            z = features[np.asarray(inds, dtype=np.int64)] if inds else np.zeros((0, 768), dtype=np.float32)
            raw_cos = z @ norm(sv) if len(z) else np.zeros(0, dtype=np.float32)
            raw_choice = int(np.argmax(raw_cos)) if len(raw_cos) else 0
            raw_idx = None if not len(inds) else int(inds[raw_choice])
            raw_iou = 0.0 if raw_idx is None else iou(box(native[raw_idx].get("bbox_xyxy")), gt.get(str(detail.get("row_key"))))
            selected.append({"row_key": str(detail.get("row_key")), "video_id": image_key[0], "image_id": image_key[1], "candidate_count": len(inds), "choice": None if deferred else choice, "defer": deferred, "logit_max": float(np.max(logits)) if len(logits) else 0.0, "selected_native_index": native_index, "selected_iou": float(selected_iou), "raw_source_mean_choice": raw_idx, "raw_source_mean_iou": float(raw_iou), "q0_reliable": bool(detail.get("q0_reliable", False))})
        records.append({"event_key": str(event.get("event_key")), "model_event_uid": str(event.get("model_event_uid")), "fold": fold, "model_fold": fold if fold in models else fold % MODEL_FOLD_COUNT, "polarity": str(event.get("polarity")), "prefix": int(event.get("prefix", 0)), "source_tracklet_key": source_key, "target_tracklet_key": str(event.get("target_tracklet_key")), "source_reliable_frozen": bool(event.get("source_reliable", False)), "target_reliable_frozen": bool(event.get("target_reliable", False)), "both_reliable_frozen": bool(event.get("both_reliable", False)), "target_rows": selected, "selected_candidate": bool(any(x["choice"] is not None for x in selected)), "selected_reliable": bool(any(x["selected_iou"] >= 0.5 for x in selected)), "raw_selected_reliable": bool(any(x["raw_source_mean_iou"] >= 0.5 for x in selected)), "candidate_count_total": int(sum(x["candidate_count"] for x in selected))})
    summary: list[dict[str, Any]] = []
    for p in PREFIXES:
        for pol in ("positive", "negative"):
            rs = [r for r in records if r["prefix"] == p and r["polarity"] == pol]
            summary.append({"prefix": p, "polarity": pol, "events": len(rs), "selected_candidate_events": sum(r["selected_candidate"] for r in rs), "selected_reliable_events": sum(r["selected_reliable"] for r in rs), "raw_source_mean_reliable_events": sum(r["raw_selected_reliable"] for r in rs), "frozen_source_reliable": sum(r["source_reliable_frozen"] for r in rs), "frozen_target_reliable": sum(r["target_reliable_frozen"] for r in rs), "frozen_both_reliable": sum(r["both_reliable_frozen"] for r in rs)})
    model_meta = {str(k): {"path": v["path"], "sha256": v["sha256"]} for k, v in models.items()}
    strategy = "frozen B84S native candidate-set listwise selector; event source track supplies same-space causal mean/prototypes; one frozen fold checkpoint per event fold"
    if RAW_ANCHOR:
        strategy = "B84S-RA fixed raw source-mean cosine plus bounded 0.05 tanh-normalized B84S-Q candidate residual; model DEFER falls back to raw candidate"
    aggregate = {"schema_version": "trackocd.phase84.b84s.event_replay.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "strategy": strategy, "raw_anchor_enabled": RAW_ANCHOR, "raw_anchor_bound": RAW_ANCHOR_BOUND if RAW_ANCHOR else None, "model_prefix": MODEL_PREFIX, "model_fold_count": MODEL_FOLD_COUNT, "records": records, "summary": summary, "model_checkpoints": model_meta, "inputs": {"native": str(NATIVE_PATH.resolve()), "native_sha256": sha(NATIVE_PATH), "native_features": str(FEATURE_PATH.resolve()), "native_features_sha256": sha(FEATURE_PATH), "b4_candidate_sets": str(B4_PATH.resolve()), "b4_candidate_sets_sha256": sha(B4_PATH), "source_cache": str(SOURCE_CACHE.resolve()), "source_cache_sha256": sha(SOURCE_CACHE), "model_manifest": str(MODEL_MANIFEST.resolve()), "model_manifest_sha256": sha(MODEL_MANIFEST), "observability": str(OBS_PATH.resolve()), "observability_sha256": sha(OBS_PATH)}, "denominators": {"positive_events": 76, "negative_events": 76, "prefixes": list(PREFIXES)}, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "event_labels_posthoc_only": True, "controller_run": False}
    return aggregate, {"schema_version": "trackocd.phase84.b84s.formal_aggregate.v1"}


def aggregate_formal() -> dict[str, Any]:
    rows = []
    for fold in range(MODEL_FOLD_COUNT):
        p = BASE / "metrics" / f"{MODEL_PREFIX}_f{fold}.json"
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    def w(key: str, section: str) -> float:
        metrics_key = f"{section}_metrics"
        den = sum(float(r[metrics_key].get("target_candidate_groups" if key == "candidate_top1_recall" else "groups", 0.0)) for r in rows)
        if key == "candidate_top1_recall":
            vals = [(r[metrics_key][key], r[metrics_key]["target_candidate_groups"]) for r in rows]
        else:
            vals = [(r[metrics_key][key], r[metrics_key]["groups"]) for r in rows]
        return float(sum(v * n for v, n in vals) / max(den, 1.0))
    # Fold metrics are intentionally schema-tolerant: the balanced B84S-Q
    # trainer does not emit the optional mean_nll field that the older B84S
    # trainer used.  Aggregate only fields present in every completed fold,
    # rather than failing before the frozen event replay is written.
    requested_metrics = ["candidate_top1_recall", "candidate_top5_recall", "defer_recall", "candidate_or_defer_accuracy", "mean_nll"]
    metrics = [k for k in requested_metrics if all(k in r["validation_metrics"] for r in rows)]
    aggregate = {"schema_version": "trackocd.phase84.b84s.formal_aggregate.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "tag": MODEL_PREFIX, "folds": rows, "validation_weighted": {k: w(k, "validation") for k in metrics}, "validation_macro": {k: float(np.mean([r["validation_metrics"][k] for r in rows])) for k in metrics}, "fit_weighted": {k: float(np.average([r["fit_metrics"][k] for r in rows], weights=[r["fit_metrics"]["groups"] for r in rows])) for k in metrics}, "formal_protocol": {"epochs": 15, "folds": MODEL_FOLD_COUNT, "candidate_action_space": "native candidates + explicit DEFER", "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}}
    return aggregate


def main() -> None:
    global MODEL_PREFIX, MODEL_FOLD_COUNT, MODEL_MANIFEST, OUTPUT_SUFFIX, RAW_ANCHOR, RAW_ANCHOR_BOUND
    ap = argparse.ArgumentParser(); ap.add_argument("--model-prefix", default=MODEL_PREFIX); ap.add_argument("--fold-count", type=int, default=MODEL_FOLD_COUNT); ap.add_argument("--suffix", default=""); ap.add_argument("--manifest", default=str(MODEL_MANIFEST)); ap.add_argument("--raw-anchor", action="store_true"); ap.add_argument("--raw-anchor-bound", type=float, default=RAW_ANCHOR_BOUND); a = ap.parse_args(); MODEL_PREFIX, MODEL_FOLD_COUNT, OUTPUT_SUFFIX, MODEL_MANIFEST, RAW_ANCHOR, RAW_ANCHOR_BOUND = a.model_prefix, a.fold_count, a.suffix, Path(a.manifest), bool(a.raw_anchor), float(a.raw_anchor_bound)
    formal = aggregate_formal()
    event, _ = evaluate()
    formal_path = OUT / f"metrics/b84s_formal_aggregate{OUTPUT_SUFFIX}.json"; event_path = OUT / f"metrics/b84s_event_replay{OUTPUT_SUFFIX}.json"; done_path = OUT / f"completion/b84s_event_replay{OUTPUT_SUFFIX}.done"
    atomic_json(formal_path, formal)
    atomic_json(event_path, event)
    atomic_json(done_path, {"status": "DONE", "formal_aggregate": str(formal_path.resolve()), "event_replay": str(event_path.resolve()), "event_replay_sha256": sha(event_path)})
    print(json.dumps({"formal_validation_weighted": formal["validation_weighted"], "p16": [x for x in event["summary"] if x["prefix"] == 16]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
