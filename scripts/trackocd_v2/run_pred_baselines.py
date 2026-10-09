#!/usr/bin/env python3
"""Run fixed baselines on the complete public predicted-track stream.

The causal pass iterates over every public predicted track and reads only its
v2 visual feature.  The evaluator-only temporal-IoU join is built and used
only after the public decisions have been sealed in per-prefix JSONL files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.evaluation.persistent import evaluate_persistent  # noqa: E402
from src.trackocd_v2.evaluation.standard_ocd import evaluate_standard  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.methods.dpmeans import OnlineDPMeans  # noqa: E402
from src.trackocd_v2.methods.nearest_prototype import NearestPrototype  # noqa: E402
from src.trackocd_v2.methods.phe_track import PHETrackAdapter  # noqa: E402
from src.trackocd_v2.protocol import load_ids  # noqa: E402
from src.trackocd_v2.sharded_cache import ShardedFeatureStore  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
TRAIN_MANIFEST = OUTPUT_TARGET / "manifests/tao_train_gt_tracks.jsonl"
TRAIN_LABELS = OUTPUT_TARGET / "manifests/private_tao_train_gt_track_labels.jsonl"
VAL_MANIFEST = OUTPUT_TARGET / "manifests/tao_val_gt_tracks.jsonl"
VAL_LABELS = OUTPUT_TARGET / "manifests/private_tao_val_gt_track_labels.jsonl"
TRAIN_FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks/train"
JOIN_AUDIT = OUTPUT_TARGET / "audit/predicted_evaluator_join.json"
JOIN_PATH = OUTPUT_TARGET / "manifests/tao_val_predicted_evaluator_join.jsonl"
SELECTION_AUDIT = OUTPUT_TARGET / "audit/frontend_selection.json"
FRONTEND_STAGE_NAMES = {
    "SimOWT/Q0": "simowt",
    "OVTR-native": "ovtr",
    "COVTrack-native": "covtrack_native",
    "COVTrack-NoSemantic": "covtrack_nosem",
}
DECISION_ROOT = OUTPUT_TARGET / "diagnostics"
TABLE_ROOT = OUTPUT_TARGET / "tables"
ROLE_ROOT = ROOT / "data/tao_ow_ocd_v1/splits"
PHE_CHECKPOINT = ROOT / "runs/phe_track/dinov2_seed1027/checkpoint.pth"


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.zeros_like(value)


def _load_prototypes() -> dict[int, dict[int, np.ndarray]]:
    labels = {str(row["sample_key"]): row for row in _read_jsonl(TRAIN_LABELS)}
    groups: dict[int, defaultdict[int, list[np.ndarray]]] = {
        prefix: defaultdict(list) for prefix in PREFIXES
    }
    for row in _read_jsonl(TRAIN_MANIFEST):
        key = str(row["sample_key"])
        label = labels.get(key)
        if label is None:
            raise ValueError(f"training label sidecar missing {key}")
        if str(label["gt_split"]) != "old":
            continue
        feature_path = TRAIN_FEATURE_ROOT / f"{key}.json"
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        for prefix in PREFIXES:
            vector = np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32)
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError(f"invalid train prototype feature for {key}, p{prefix}")
            groups[prefix][int(label["gt_category_id"])].append(vector)
    return {
        prefix: {category: _unit(np.mean(values, axis=0)) for category, values in by_category.items()}
        for prefix, by_category in groups.items()
    }


def _make_method(method_name: str, prototypes: dict[int, dict[int, np.ndarray]], prefix: int, radius: int, device: str) -> Any:
    if method_name == "nearest":
        return NearestPrototype(known_prototypes=prototypes[prefix])
    if method_name == "dpmeans":
        return OnlineDPMeans(known_prototypes=prototypes[prefix])
    if method_name == "phe":
        if not PHE_CHECKPOINT.exists():
            raise FileNotFoundError(PHE_CHECKPOINT)
        return PHETrackAdapter(PHE_CHECKPOINT, radius=radius, device=device)
    raise ValueError(method_name)


def _make_methods(method_name: str, prototypes: dict[int, dict[int, np.ndarray]], radius: int, device: str) -> dict[int, Any]:
    """Compatibility factory for the small GT/Test replay routes."""

    return {
        prefix: _make_method(method_name, prototypes, prefix, radius, device)
        for prefix in PREFIXES
    }


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def _resolve_causal_stream() -> tuple[Path, Any, dict[str, Any]] | None:
    """Resolve the current causal stream and its representation source.

    A selected frontend must use its completed formal sharded cache.  The
    retained pre-selection JSON cache belongs only to the bounded sanity route
    and is never accepted by this formal baseline runner.
    """

    selection = json.loads(SELECTION_AUDIT.read_text(encoding="utf-8")) if SELECTION_AUDIT.exists() else {}
    selected = str(selection.get("selected_frontend") or "")
    if selection.get("status") == "FINAL_PHYSICAL_FRONTEND_SELECTED":
        slug = FRONTEND_STAGE_NAMES.get(selected)
        if slug is None:
            raise ValueError(f"unknown selected frontend: {selected}")
        representation_path = OUTPUT_TARGET / "audit/representation_decision.json"
        if not representation_path.is_file():
            return None
        representation = json.loads(representation_path.read_text(encoding="utf-8"))
        if (
            representation.get("status") != "FINAL_REPRESENTATION_SELECTED"
            or representation.get("formal_common_feature_cache_authorized") is not True
            or representation.get("final_physical_frontend") != selected
        ):
            return None
        stage_path = OUTPUT_TARGET / "audit" / f"frontend_{slug}.json"
        stage = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.exists() else {}
        normalized = stage.get("normalized_outputs") or {}
        stream = Path(str(normalized.get("physical_stream", ""))).resolve()
        manifest = OUTPUT_TARGET / "features/formal" / slug / "cache_manifest.json"
        if not stream.is_file() or not manifest.is_file():
            return None
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE" or payload.get("frontend") != selected:
            return None
        store = ShardedFeatureStore(manifest)
        return stream, store, {
            "kind": "formal_sharded",
            "frontend": selected,
            "frontend_slug": slug,
            "stream": str(stream),
            "cache_manifest": str(manifest.resolve()),
            "decision_namespace": f"frontend_{slug}",
        }
    return None


def _decision_evidence_paths(path: Path) -> tuple[Path, Path]:
    return (
        path.with_name(path.name + ".launched"),
        path.with_name(path.name + ".done"),
    )


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _claim_decision_prefix(path: Path, method_name: str, prefix: int, stream_sha256: str) -> Path:
    launched, _done = _decision_evidence_paths(path)
    if launched.exists():
        try:
            payload = json.loads(launched.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid decision launch marker: {launched}") from exc
        if _pid_is_alive(pid):
            raise RuntimeError(f"decision prefix is already owned by live PID {pid}: {path}")
        launched.unlink()
    launched.parent.mkdir(parents=True, exist_ok=True)
    try:
        with launched.open("x", encoding="utf-8") as handle:
            json.dump({
                "pid": os.getpid(),
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "method": method_name,
                "prefix": int(prefix),
                "stream_sha256": stream_sha256,
            }, handle, sort_keys=True)
    except FileExistsError as exc:
        raise RuntimeError(f"decision prefix claim raced: {launched}") from exc
    return launched


def _completed_decision_prefix(path: Path, method_name: str, prefix: int, stream_sha256: str) -> tuple[bool, int]:
    _launched, done = _decision_evidence_paths(path)
    if not path.is_file() or not done.is_file():
        return False, 0
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "COMPLETE"
            or payload.get("method") != method_name
            or int(payload.get("prefix")) != int(prefix)
            or payload.get("stream_sha256") != stream_sha256
        ):
            return False, 0
        count = int(payload["decision_count"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False, 0
    return count >= 0, count


def _public_decisions(
    method_name: str,
    prototypes: dict[int, dict[int, np.ndarray]],
    stream_manifest: Path,
    feature_store: Any,
    source: dict[str, Any],
    *,
    radius: int,
    device: str,
) -> tuple[dict[int, Path], int]:
    """Seal one causal prefix at a time and resume completed prefixes.

    Prefixes are independent causal replays.  Serializing them keeps only one
    anonymous-state memory resident, while the per-prefix evidence prevents a
    later retry from replaying already sealed decisions.  The decision logic,
    stream order, feature inputs, and thresholds are unchanged.
    """

    DECISION_ROOT.mkdir(parents=True, exist_ok=True)
    namespace = str(source["decision_namespace"])
    final_paths = {prefix: DECISION_ROOT / f"{namespace}_{method_name}_decisions_p{prefix}.jsonl" for prefix in PREFIXES}
    stream_sha256 = sha256_file(stream_manifest)
    counts: list[int] = []
    for prefix in PREFIXES:
        final_path = final_paths[prefix]
        complete, completed_count = _completed_decision_prefix(final_path, method_name, prefix, stream_sha256)
        if complete:
            counts.append(completed_count)
            continue

        _claim_decision_prefix(final_path, method_name, prefix, stream_sha256)
        temporary_path = _temporary(final_path)
        method = _make_method(method_name, prototypes, prefix, radius, device)
        count = 0
        previous_order = -1
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                for row in _read_jsonl(stream_manifest):
                    key = str(row["sample_key"])
                    stream_order = int(row["stream_order"])
                    if stream_order < previous_order:
                        raise ValueError(f"causal stream order regressed at {key}: {stream_order} < {previous_order}")
                    previous_order = stream_order
                    feature = feature_store.get(key)
                    if set(feature) & {"gt_category_id", "gt_split", "gt_match_id", "category_id", "category_name", "text"}:
                        raise ValueError(f"causal feature contains evaluator/semantic fields: {key}")
                    if str(feature.get("source_split")) not in {"pred", "val_predicted"}:
                        raise ValueError(f"predicted decision read non-predicted feature: {key}")
                    vector = np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32)
                    if vector.shape != (768,) or not np.isfinite(vector).all():
                        raise ValueError(f"invalid predicted feature for {key}, p{prefix}")
                    decision = dict(method.step(vector))
                    handle.write(json.dumps({
                        "sample_key": key,
                        "stream_order": stream_order,
                        "decision": decision,
                    }, sort_keys=True, separators=(",", ":")) + "\n")
                    count += 1
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, final_path)
            _launched, done_path = _decision_evidence_paths(final_path)
            atomic_json(done_path, {
                "schema_version": "trackocd.v2.predicted_decision_prefix.v1",
                "status": "COMPLETE",
                "method": method_name,
                "prefix": int(prefix),
                "stream_sha256": stream_sha256,
                "decision_count": int(count),
                "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            })
            _launched.unlink(missing_ok=True)
            counts.append(count)
        except Exception:
            # Keep the launch marker and any temporary output as recovery
            # evidence.  A later invocation reclaims the marker after this PID
            # exits and reruns only this unsealed prefix.
            raise
    if len(set(counts)) != 1:
        raise RuntimeError(f"causal prefix decision counts disagree: {counts}")
    return final_paths, counts[0]


def _ensure_evaluator_join(source: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    if source["kind"] == "formal_sharded":
        slug = str(source["frontend_slug"])
        join_audit = OUTPUT_TARGET / "audit" / f"frontend_{slug}_evaluator_join.json"
        join_path = OUTPUT_TARGET / "manifests" / f"frontend_{slug}_evaluator_join.jsonl"
        physical_path = Path(str(source["stream"])).resolve()
        if join_audit.exists() and join_path.exists():
            audit = json.loads(join_audit.read_text(encoding="utf-8"))
            if (
                audit.get("status") == "COMPLETE"
                and audit.get("source_physical_stream_sha256") == sha256_file(physical_path)
                and audit.get("decisions_sealed_before_join") is True
            ):
                return audit, join_path
        command = [
            sys.executable,
            str((ROOT / "scripts/trackocd_v2/build_frontend_evaluator_join.py").resolve()),
            "--physical-stream", str(physical_path),
            "--frontend", slug,
            "--join-path", str(join_path.resolve()),
            "--audit-path", str(join_audit.resolve()),
            "--decisions-sealed",
        ]
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"frontend evaluator-only join failed: {result.returncode}")
        return json.loads(join_audit.read_text(encoding="utf-8")), join_path

    if JOIN_AUDIT.exists() and JOIN_PATH.exists():
        audit = json.loads(JOIN_AUDIT.read_text(encoding="utf-8"))
        if (
            audit.get("status") == "COMPLETE"
            and audit.get("public_prediction_source_sha256") == sha256_file(ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream.jsonl")
        ):
            return audit, JOIN_PATH
    command = [sys.executable, str(ROOT / "scripts/trackocd_v2/build_predicted_evaluator_join.py")]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"evaluator-only predicted join failed: {result.returncode}")
    return json.loads(JOIN_AUDIT.read_text(encoding="utf-8")), JOIN_PATH


def _load_join(path: Path) -> list[dict]:
    return list(_read_jsonl(path))


def _load_fixed_gt_rows() -> list[dict]:
    """Load all non-distractor GT targets for the fixed predicted denominator.

    The rows are evaluator-only.  A missing physical prediction is represented
    later as a DEFER decision, never removed from the GT target population.
    """

    labels = {str(row["sample_key"]): row for row in _read_jsonl(VAL_LABELS)}
    rows = []
    for row in _read_jsonl(VAL_MANIFEST):
        key = str(row["sample_key"])
        label = labels.get(key)
        if label is None:
            raise ValueError(f"validation GT label sidecar missing {key}")
        if str(label.get("gt_split")) == "distractor" or bool(label.get("is_distractor", False)):
            continue
        rows.append({
            "gt_sample_key": key,
            "gt_physical_track_id": str(row["physical_track_id"]),
            "video_id": int(row["video_id"]),
            "gt_stream_order": int(row.get("stream_order", len(rows))),
            "gt_category_id": int(label["gt_category_id"]),
            "gt_split": str(label["gt_split"]),
            "evaluator_track_key": key,
        })
    if not rows:
        raise ValueError("validation GT denominator is empty")
    return rows


def _fixed_denominator_rows(join_rows: list[dict], gt_rows: list[dict]) -> tuple[list[dict], int]:
    """Merge matched causal rows with unmatched GT rows without model leakage."""

    matched_by_gt: dict[str, dict] = {}
    for row in join_rows:
        key = str(row["gt_sample_key"])
        if key in matched_by_gt:
            raise ValueError(f"duplicate evaluator match for GT target {key}")
        matched_by_gt[key] = dict(row)

    matched = []
    missing = []
    for gt in gt_rows:
        key = str(gt["gt_sample_key"])
        row = matched_by_gt.get(key)
        if row is not None:
            row["gt_stream_order"] = int(gt["gt_stream_order"])
            row["evaluator_track_key"] = key
            row["causal_decision_available"] = True
            matched.append(row)
            continue
        # Missing physical observations are an evaluator-side DEFER.  They do
        # not receive a synthetic physical ID or a causal feature.
        missing.append({
            **gt,
            "sample_key": f"__UNMATCHED_GT__/{key}",
            "source_sample_id": None,
            "physical_track_id": None,
            "predicted_physical_track_id": None,
            "stream_order": None,
            "observation_status": "UNMATCHED_GT_TARGET",
            "causal_decision_available": False,
            "evaluator_only": True,
        })
    matched.sort(key=lambda row: (int(row["stream_order"]), int(row["video_id"]), str(row["sample_key"])))
    missing.sort(key=lambda row: (int(row["gt_stream_order"]), int(row["video_id"]), str(row["gt_sample_key"])))
    return matched + missing, len(missing)


def _load_matched_decisions(path: Path, wanted: set[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in _read_jsonl(path):
        key = str(row["sample_key"])
        if key in wanted:
            if key in result:
                raise ValueError(f"duplicate decision for {key}: {path}")
            result[key] = row["decision"]
    missing = wanted - result.keys()
    if missing:
        raise ValueError(f"decision file misses {len(missing)} evaluator keys: {sorted(missing)[:5]}")
    return result


def _slim_standard(metric: dict) -> dict:
    keep = ("old_acc", "new_acc", "h_score", "all_acc", "old_correct", "old_total", "new_correct", "new_total", "state_count", "hungarian_mapping")
    return {key: metric[key] for key in keep}


def _evaluate(decision_paths: dict[int, Path], join_rows: list[dict], gt_rows: list[dict], known_ids: set[int], novel_ids: set[int], distractor_ids: set[int]) -> tuple[dict, int]:
    evaluator_rows, missing_count = _fixed_denominator_rows(join_rows, gt_rows)
    wanted = {str(row["sample_key"]) for row in evaluator_rows if row.get("causal_decision_available")}
    result: dict[str, dict] = {}
    for prefix in PREFIXES:
        decision_by_key = _load_matched_decisions(decision_paths[prefix], wanted)
        decisions = [
            decision_by_key[str(row["sample_key"])] if row.get("causal_decision_available")
            else {"kind": "DEFER", "token": None, "observation_status": "UNMATCHED_GT_TARGET"}
            for row in evaluator_rows
        ]
        standard = evaluate_standard(
            evaluator_rows,
            decisions,
            known_ids=known_ids,
            novel_ids=novel_ids,
            distractor_ids=distractor_ids,
        )
        persistent = evaluate_persistent(
            evaluator_rows,
            decisions,
            known_ids=known_ids,
            novel_ids=novel_ids,
            distractor_ids=distractor_ids,
        )
        result[str(prefix)] = {
            "standard": _slim_standard(standard),
            "persistent": persistent,
        }
    return result, missing_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("nearest", "dpmeans", "phe"), required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out = ensure_output_layout()
    table_path = TABLE_ROOT / f"pred_{args.method}.json"
    resolved = _resolve_causal_stream()
    if resolved is None:
        atomic_json(table_path, {
            "schema_version": "trackocd.v2.pred_baseline.v1",
            "status": "WAITING_FEATURES",
            "method": args.method,
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reason": "Formal predicted baselines wait for a selected frontend and its completed sharded cache; the retained pre-selection JSON cache is sanity-only.",
            "formal_common_feature_cache_authorized": False,
        })
        return 2
    stream_manifest, feature_store, source = resolved
    for path in (stream_manifest, TRAIN_MANIFEST, TRAIN_LABELS, VAL_MANIFEST, VAL_LABELS):
        if not path.exists():
            raise FileNotFoundError(path)

    prototypes = _load_prototypes()
    decision_paths, decision_count = _public_decisions(
        args.method,
        prototypes,
        stream_manifest,
        feature_store,
        source,
        radius=args.radius,
        device=args.device,
    )
    join_audit, join_path = _ensure_evaluator_join(source)
    join_rows = _load_join(join_path)
    gt_rows = _load_fixed_gt_rows()
    join_audit_path = (
        OUTPUT_TARGET / "audit" / f"frontend_{source['frontend_slug']}_evaluator_join.json"
        if source["kind"] == "formal_sharded"
        else JOIN_AUDIT
    )
    known_ids = load_ids(ROLE_ROOT / "known_ids.json")
    novel_ids = load_ids(ROLE_ROOT / "unknown_ids_val.json")
    distractor_ids = load_ids(ROLE_ROOT / "distractor_ids.json")
    prefixes, missing_gt_targets = _evaluate(decision_paths, join_rows, gt_rows, known_ids, novel_ids, distractor_ids)
    aggregate = {
        prefix: {
            "standard_mean": {
                name: float(prefixes[prefix]["standard"][name])
                for name in ("old_acc", "new_acc", "h_score", "all_acc")
            },
            "persistent_mean": {
                name: float(prefixes[prefix]["persistent"][name])
                for name in ("commit_ct", "false_assignment_rate")
            },
        }
        for prefix in prefixes
    }
    result = {
        "schema_version": "trackocd.v2.pred_baseline.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": args.method,
        "prefixes": list(PREFIXES),
        "fixed_method_parameters": {
            "nearest": {"tau_known": 0.35, "tau_existing": 0.55},
            "dpmeans": {"lambda_distance": 0.45, "tau_known": 0.35},
            "phe": {"radius": args.radius, "checkpoint": str(PHE_CHECKPOINT.resolve())},
        }[args.method],
        "train_known_prototype_categories_by_prefix": {
            str(prefix): sorted(int(key) for key in prototypes[prefix]) for prefix in PREFIXES
        },
        "public_stream": {
            "manifest": str(stream_manifest.resolve()),
            "manifest_sha256": sha256_file(stream_manifest),
            "decision_count": decision_count,
            "decision_files": {str(prefix): str(path.resolve()) for prefix, path in decision_paths.items()},
            "all_public_tracks_received_a_causal_decision": True,
            "source_kind": source["kind"],
            "frontend": source.get("frontend"),
            "feature_source": source.get("cache_manifest", source.get("feature_audit")),
        },
        "evaluator_subset": {
            "join_audit": str(join_audit_path.resolve()),
            "join_audit_sha256": sha256_file(join_audit_path),
            "join_rows": len(join_rows),
            "fixed_gt_denominator_rows": len(gt_rows),
            "unmatched_gt_targets_counted_as_defer": missing_gt_targets,
            "fixed_gt_denominator": True,
            "public_decisions_filtered_only_after_causal_pass": True,
            "gt_matching_used_as_model_input": False,
            "historical_matched_stream_used_for_inference": False,
            "matching_source": join_audit.get("historical_diagnostic_source", join_audit.get("source_physical_stream")),
        },
        "prefixes_result": prefixes,
        "aggregate": aggregate,
        "test_semantic_accessed": False,
        "model_input_contract": {
            "input_fields": ["DINOv2 prefix feature", "causal physical-stream order"],
            "gt_category_id": False,
            "gt_split": False,
            "temporal_iou_match": False,
            "text_or_category_logits": False,
        },
    }
    atomic_json(table_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
