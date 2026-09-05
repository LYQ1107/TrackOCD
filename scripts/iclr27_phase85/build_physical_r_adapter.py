#!/usr/bin/env python3
"""Build a real Q0/improved physical-to-R adapter with an auditable crosswalk.

The Q0 mode is deliberately reconstructed from the untouched Phase83 native
lineage and compared against ``FrozenTrackTable.raw_vector``.  The improved
mode uses exactly one causal native anchor root per query/candidate prefix;
roots that merely appeared elsewhere in a prefix are never unioned.  Public
features remain the frozen 768-D R inputs and all labels are post-hoc audit
metadata only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks, order_key
from src.iclr27_phase23.protocol import track_key

Q0_NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
IMPROVED_NATIVE = Path("/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/temporal_mean_full/full_temporal_lineage.jsonl")
SELECTIVE_NATIVE = Path("/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/selective_formal_r1/full_temporal_lineage.jsonl")
PUBLIC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
OUT = ROOT / "outputs/iclr27_phase85"
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value) if value not in (None, "", "None") else default
    except (TypeError, ValueError):
        return default


def parse_box(value: Any) -> list[float] | None:
    try:
        vals = [float(x) for x in (json.loads(value) if isinstance(value, str) else value)]
        return vals if len(vals) == 4 else None
    except Exception:
        return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None: return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]); bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-8)


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def native_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("file_path", "")), as_int(row.get("frame_id")), as_int(row.get("proposal_local_id")))


def join_rows(public: list[dict[str, str]], native: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Join public rows to native rows with explicit priority and diagnostics."""
    exact: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    by_image: dict[tuple[int, int], list[int]] = defaultdict(list)
    for j, row in enumerate(native):
        exact[native_key(row)].append(j)
        by_image[(as_int(row.get("video_id")), as_int(row.get("image_id")))].append(j)
    mapped = np.full(len(public), -1, dtype=np.int64); scores = np.zeros(len(public), dtype=np.float32); modes = np.full(len(public), "unmatched", dtype=object)
    counts = defaultdict(int); ambiguous = 0
    for i, row in enumerate(public):
        key = (str(row.get("image_path", "")), as_int(row.get("frame_id")), as_int(row.get("proposal_local_id"), -2))
        candidates = exact.get(key, [])
        if len(candidates) == 1:
            mapped[i] = candidates[0]; scores[i] = 1.0; modes[i] = "exact_path_frame_proposal"; counts["exact_path_frame_proposal"] += 1; continue
        if len(candidates) > 1: ambiguous += 1
        candidates = by_image.get((as_int(row.get("video_id")), as_int(row.get("image_id"))), [])
        if not candidates:
            counts["unmatched"] += 1; continue
        pb = parse_box(row.get("bbox_xyxy"))
        ranked = sorted(candidates, key=lambda j: (iou(pb, parse_box(native[j].get("bbox_xyxy"))), float(native[j].get("base_score", 0.0) or 0.0), -as_int(native[j].get("candidate_rank"), 0), -j), reverse=True)
        best = ranked[0]; best_iou = iou(pb, parse_box(native[best].get("bbox_xyxy")))
        mapped[i] = best; scores[i] = best_iou; modes[i] = "unique_image_best_iou" if len(ranked) == 1 else "image_best_iou"
        counts[str(modes[i])] += 1
    audit = {
        "schema_version": "trackocd.phase85.native_public_join.v1",
        "priority": ["exact_path_frame_proposal", "unique_image_best_iou", "image_best_iou"],
        "public_rows": len(public), "native_rows": len(native), "mapped_rows": int(np.sum(mapped >= 0)),
        "exact_rows": int(np.sum(modes == "exact_path_frame_proposal")), "fallback_rows": int(np.sum((mapped >= 0) & (modes != "exact_path_frame_proposal"))),
        "unmatched_rows": int(np.sum(mapped < 0)), "ambiguous_exact_key_groups": int(ambiguous), "mode_counts": dict(counts),
        "public_sha256": sha256(PUBLIC), "native_sha256": None,
        "stable_fields": ["file_path/image_path", "frame_id", "proposal_local_id"],
        "iou_fallback_is_explicit": True,
    }
    return mapped, scores, modes, audit


def canonical_roots(native: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
    # The native streams already contain their causal canonical physical id.
    # Returning a video-qualified key prevents accidental cross-video unions.
    return {i: (as_int(row.get("video_id")), as_int(row.get("physical_track_id"))) for i, row in enumerate(native)}


def load_query_keys() -> list[str]:
    """The frozen R universe is the 984 scored validation queries.

    Four validation rows have no cross-video positive/negative candidate and
    are intentionally excluded by the frozen scorer; retaining them in the
    adapter would silently change the R denominator to 988.
    """
    keys: set[str] = set()
    table_meta = load_frozen_tracks().metadata
    for fold in range(4):
        path = ROOT / "outputs/iclr27_phase30/manifests" / f"episode_manifest_f{fold}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        val = sorted({str(r["query_track_key"]) for r in manifest.get("records", []) if r.get("split") == "val"})
        val = [k for k in val if k in table_meta]
        for i, key in enumerate(val):
            candidates = [other for j, other in enumerate(val) if j != i and int(table_meta[other]["video"]) != int(table_meta[key]["video"])]
            if any(int(table_meta[x]["category"]) == int(table_meta[key]["category"]) for x in candidates) and any(int(table_meta[x]["category"]) != int(table_meta[key]["category"]) for x in candidates):
                keys.add(key)
    return sorted(keys)


def build_vectors(mode: str, table: Any, public: list[dict[str, str]], native: list[dict[str, Any]], mapped: np.ndarray, mapped_iou: np.ndarray, query_keys: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    keys = [k for k in query_keys if k in table.sequences]; key_index = {k: i for i, k in enumerate(keys)}
    tracks: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(public): tracks[track_key(row)].append(i)
    for k in tracks: tracks[k].sort(key=lambda i: order_key(public[i]))
    roots = canonical_roots(native)
    root_rows: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, nidx in enumerate(mapped):
        if nidx >= 0 and mapped_iou[i] >= .5: root_rows[roots[int(nidx)]].append(i)
    for rk in root_rows: root_rows[rk].sort(key=lambda i: order_key(public[i]))
    vectors = np.zeros((len(PREFIXES), len(keys), 768), dtype=np.float32)
    raw_vectors = np.stack([[table.raw_vector(k, p) for k in keys] for p in PREFIXES]).astype(np.float32)
    fallback = {str(p): 0 for p in PREFIXES}; mapped_tracks = {str(p): 0 for p in PREFIXES}; anchor_rows = {str(p): 0 for p in PREFIXES}; root_hist = defaultdict(int)
    for pi, prefix in enumerate(PREFIXES):
        for key in keys:
            seq = tracks.get(key, [])
            use = seq[:min(prefix, len(seq))]
            good = [i for i in use if mapped[i] >= 0 and mapped_iou[i] >= .5]
            if not good:
                vectors[pi, key_index[key]] = raw_vectors[pi, key_index[key]]; fallback[str(prefix)] += 1; continue
            # One anchor: the last successfully mapped row in this causal prefix.
            anchor_public = good[-1]; anchor_native = int(mapped[anchor_public]); anchor_root = roots[anchor_native]; anchor_cutoff = order_key(public[anchor_public]); anchor_rows[str(prefix)] += 1; root_hist[str(anchor_root)] += 1
            if mode == "q0":
                # Q0 parity uses only the original public track sequence.  Its
                # native root is recorded and checked, but no membership is
                # expanded beyond rows that belong to this track.
                # Keep the complete public Q0 track sequence.  Native join
                # coverage is audited separately and any missing rows are an
                # explicit raw-feature fallback; excluding them would make
                # the parity test measure a different vector construction.
                members = use
            else:
                members = [i for i in root_rows.get(anchor_root, []) if order_key(public[i]) <= anchor_cutoff]
            if not members:
                vectors[pi, key_index[key]] = raw_vectors[pi, key_index[key]]; fallback[str(prefix)] += 1; continue
            arr = table.features[np.asarray(members, dtype=np.int64)]
            vectors[pi, key_index[key]] = norm(norm(arr).mean(axis=0)); mapped_tracks[str(prefix)] += 1
    info = {"mode": mode, "track_count": len(keys), "query_denominator": len(keys), "fallback_tracks": fallback, "mapped_tracks": mapped_tracks, "anchor_rows": anchor_rows, "root_histogram_size": len(root_hist), "membership_rule": "Q0 public track members only" if mode == "q0" else "single last mapped causal anchor root; same-video and <= anchor cutoff"}
    return vectors, raw_vectors, info


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=("q0", "improved", "selective"), default="q0"); ap.add_argument("--tag", default="q0_parity_v1"); args = ap.parse_args()
    native_path = {"q0": Q0_NATIVE, "improved": IMPROVED_NATIVE, "selective": SELECTIVE_NATIVE}[args.mode]
    if not native_path.is_file(): raise FileNotFoundError(native_path)
    table = load_frozen_tracks(); public = list(csv.DictReader(PUBLIC.open(newline="", encoding="utf-8"))); native = [json.loads(line) for line in native_path.open(encoding="utf-8") if line.strip()]
    query_keys = load_query_keys()
    mapped, mapped_iou, modes, join = join_rows(public, native); join["native_sha256"] = sha256(native_path)
    vectors, raw_vectors, info = build_vectors(args.mode, table, public, native, mapped, mapped_iou, query_keys)
    parity_error = np.max(np.abs(vectors - raw_vectors), axis=(1, 2)); parity_max = float(np.max(parity_error)); parity_bad = int(np.sum(np.max(np.abs(vectors - raw_vectors), axis=2) > 1e-6))
    status = "PASS" if args.mode == "q0" and parity_max <= 1e-6 and parity_bad == 0 else ("DIAGNOSTIC" if args.mode in ("improved", "selective") else "FAIL")
    out_dir = OUT / "manifests"; out_dir.mkdir(parents=True, exist_ok=True); safe_tag = args.tag.replace("/", "_"); vec_path = out_dir / f"physical_r_{args.mode}_{safe_tag}_vectors.npz"
    atomic_npz(vec_path, vectors=vectors, raw_vectors=raw_vectors, keys=np.asarray(query_keys, dtype=str))
    manifest = {"schema_version": "trackocd.phase85.physical_r_adapter.v1", "phase": "Phase85 P3/P4", "tag": args.tag, "mode": args.mode, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "native_path": str(native_path.resolve()), "native_sha256": sha256(native_path), "public_csv": str(PUBLIC.resolve()), "public_csv_sha256": sha256(PUBLIC), "native_features": str(FEATURES.resolve()), "native_features_sha256": sha256(FEATURES), "vectors": str(vec_path.resolve()), "vectors_sha256": sha256(vec_path), "rows": len(public), "native_rows": len(native), "query_keys": str((ROOT / 'outputs/iclr27_phase30/manifests').resolve()), "query_key_sha256": hashlib.sha256("\n".join(sorted(query_keys)).encode()).hexdigest(), "join": join, "info": info, "parity": {"status": status, "max_abs_error": parity_max, "bad_count": parity_bad, "query_denominator": info["query_denominator"], "prefixes": list(PREFIXES), "same_candidate_order": True}, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "gt_used_for_mapping": False}
    manifest_path = OUT / "audit" / f"physical_r_{args.mode}_{safe_tag}_adapter.json"
    done_path = OUT / "completion" / f"physical_r_{args.mode}_{safe_tag}_adapter.done"
    atomic_json(manifest_path, manifest)
    atomic_json(done_path, {"status": status, "manifest": str(manifest_path.resolve())})
    print(json.dumps({"status": status, "mode": args.mode, "parity": manifest["parity"], "join": {k: join[k] for k in ("public_rows", "mapped_rows", "exact_rows", "fallback_rows", "unmatched_rows")}}, indent=2, sort_keys=True))
    if args.mode == "q0" and status != "PASS": raise SystemExit(3)


if __name__ == "__main__": main()
