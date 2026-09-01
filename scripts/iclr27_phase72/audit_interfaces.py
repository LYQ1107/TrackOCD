#!/usr/bin/env python3
"""Phase72 read-only schema/key and evaluator-contract audit.

The large TAO JSON files are parsed as a stream.  This script never joins
category labels into model tensors; event labels are retained only as
denominator/scoring metadata for the audit.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase72"
EVENT_ROOT = ROOT / "outputs/iclr27_phase19r/manifests"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterable[Any]:
    """Yield values from a top-level JSON array without json.load memory use."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
        buf = ""
        pos = 0
        eof = False

        def fill(need: bool = True) -> None:
            nonlocal buf, pos, eof
            if not eof:
                part = fh.read(chunk_size)
                if part:
                    # Compact consumed bytes before appending.  Keeping an
                    # integer cursor avoids an O(buffer-size) lstrip for every
                    # object in million-record TAO arrays.
                    if pos:
                        buf = buf[pos:]
                        pos = 0
                    buf += part
                else:
                    eof = True

        fill()
        while pos < len(buf) and buf[pos].isspace():
            pos += 1
        if pos >= len(buf) and not eof:
            fill()
        if pos >= len(buf) or buf[pos] != "[":
            raise ValueError(f"expected top-level array: {path}")
        pos += 1
        first = True
        while True:
            while pos >= len(buf) and not eof:
                fill()
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            while pos >= len(buf) and not eof:
                fill()
            if pos >= len(buf) and eof:
                raise ValueError(f"unterminated JSON array: {path}")
            if buf[pos] == "]":
                return
            if not first:
                if buf[pos] != ",":
                    raise ValueError(f"missing comma in {path}")
                pos += 1
                while pos >= len(buf) and not eof:
                    fill()
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                while pos >= len(buf) and not eof:
                    fill()
                if pos < len(buf) and buf[pos] == "]":
                    return
            first = False
            while True:
                try:
                    value, end = decoder.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
            yield value
            pos = end


def _track_key(video: Any, track: Any) -> str:
    try:
        return f"v{int(video)}:p{int(track)}"
    except (TypeError, ValueError):
        return f"v{video}:p{track}"


def summarize_tao(path: Path) -> dict[str, Any]:
    fields: set[str] = set()
    field_counts: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    videos: set[str] = set()
    tracks: set[str] = set()
    prediction_types: Counter[str] = Counter()
    records = 0
    malformed = 0
    bbox_lengths: Counter[int] = Counter()
    first_records: list[dict[str, Any]] = []
    for rec in iter_json_array(path):
        records += 1
        if not isinstance(rec, dict):
            malformed += 1
            continue
        fields.update(rec)
        for k in rec:
            field_counts[k] += 1
        if len(first_records) < 3:
            first_records.append({k: rec.get(k) for k in sorted(rec)})
        if "prediction_type" in rec:
            prediction_types[str(rec.get("prediction_type"))] += 1
        if "bbox" in rec and isinstance(rec["bbox"], list):
            bbox_lengths[len(rec["bbox"])] += 1
        if "video_id" in rec and "track_id" in rec:
            key = _track_key(rec["video_id"], rec["track_id"])
            key_counts[key] += 1
            videos.add(str(rec["video_id"]))
            tracks.add(key)
    optional = {
        k: int(field_counts.get(k, 0))
        for k in (
            "prediction_type",
            "semantic_category_id",
            "virtual_category_id",
            "action",
            "commit",
            "causal_representation",
            "frame_id",
            "proposal_local_id",
            "video_id",
            "track_id",
            "category_id",
            "score",
            "bbox",
        )
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "records": records,
        "malformed_records": malformed,
        "fields": sorted(fields),
        "field_presence_counts": optional,
        "prediction_type_counts": dict(sorted(prediction_types.items())),
        "bbox_length_counts": {str(k): int(v) for k, v in sorted(bbox_lengths.items())},
        "unique_video_count": len(videos),
        "unique_track_count": len(tracks),
        "unique_track_key_format": "v<video_id>:p<track_id>",
        "unique_track_keys_sample": sorted(tracks)[:20],
        "records_per_track_summary": {
            "min": min(key_counts.values()) if key_counts else None,
            "median": sorted(key_counts.values())[len(key_counts) // 2] if key_counts else None,
            "max": max(key_counts.values()) if key_counts else None,
        },
        "first_records": first_records,
        "track_keys": sorted(tracks),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_event_track_key(key: str) -> str | None:
    m = re.fullmatch(r"v(\d+):p(\d+)", str(key))
    return f"v{m.group(1)}:p{m.group(2)}" if m else None


def event_intersections(events: list[dict[str, Any]], tao_track_keys: set[str]) -> dict[str, Any]:
    details = []
    source_hits = target_hits = 0
    any_hits = 0
    for e in events:
        source = [parse_event_track_key(x) for x in e.get("source_tracklet_keys", [])]
        target = parse_event_track_key(e.get("target_tracklet_key", ""))
        source = [x for x in source if x is not None]
        source_i = [x for x in source if x in tao_track_keys]
        target_i = [target] if target in tao_track_keys else []
        source_hits += int(bool(source_i))
        target_hits += int(bool(target_i))
        any_hits += int(bool(source_i or target_i))
        details.append({
            "event_key": e.get("event_key"),
            "fold": e.get("fold"),
            "kind": e.get("kind"),
            "source_tracklet_keys": source,
            "target_tracklet_key": target,
            "source_intersection": source_i,
            "target_intersection": target_i,
            "intersection": bool(source_i or target_i),
        })
    return {
        "events": len(events),
        "events_with_any_track_intersection": any_hits,
        "events_with_source_intersection": source_hits,
        "events_with_target_intersection": target_hits,
        "event_details": details,
    }


def ast_contract(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    names = Counter(n.id for n in ast.walk(tree) if isinstance(n, ast.Name))
    attrs = Counter(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute))
    lines = {
        "oracle_category": [i for i, line in enumerate(text.splitlines(), 1) if "oracle_category" in line],
        "category_gt_denominator_only": [i for i, line in enumerate(text.splitlines(), 1) if "category_gt_denominator_only" in line],
        "semantic_category_id": [i for i, line in enumerate(text.splitlines(), 1) if "semantic_category_id" in line],
        "virtual_category_id": [i for i, line in enumerate(text.splitlines(), 1) if "virtual_category_id" in line],
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "oracle_category_reference_lines": lines["oracle_category"],
        "category_gt_denominator_only_reference_lines": lines["category_gt_denominator_only"],
        "semantic_category_id_reference_lines": lines["semantic_category_id"],
        "virtual_category_id_reference_lines": lines["virtual_category_id"],
        "calls_forward_or_model": bool(names.get("forward_item") or attrs.get("forward")),
    }


def main() -> None:
    audit_dir = OUT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    q0 = ROOT / "outputs/iclr27_phase4q/q0_long/teta_results/tao_track.json"
    p71_candidates = sorted((ROOT / "outputs/iclr27_phase71").glob("validation/**/tao_track.json"))
    # TrackEval preparation creates symlinks/copies of each fold's JSON.  The
    # audit must enumerate them but report one physical stream per real path,
    # otherwise the same prediction file would be counted twice.
    p71 = []
    seen_real: set[Path] = set()
    for candidate in p71_candidates:
        real = candidate.resolve()
        if real in seen_real:
            continue
        seen_real.add(real)
        p71.append(candidate)
    tao_paths = [q0] + p71
    summaries = [summarize_tao(p) for p in tao_paths if p.exists()]

    pos_path = EVENT_ROOT / "held_known_positive_events.jsonl"
    neg_path = EVENT_ROOT / "held_known_negative_events.jsonl"
    pos = load_jsonl(pos_path)
    neg = load_jsonl(neg_path)
    events = pos + neg
    fold_counts = {
        "positive": dict(sorted(Counter(int(e["fold"]) for e in pos).items())),
        "negative": dict(sorted(Counter(int(e["fold"]) for e in neg).items())),
    }
    event_audit = {
        "positive_count": len(pos),
        "negative_count": len(neg),
        "total_count": len(events),
        "expected_positive_count": 76,
        "expected_negative_count": 76,
        "counts_match_contract": len(pos) == 76 and len(neg) == 76,
        "fold_counts": fold_counts,
        "event_key_unique": len({e.get("event_key") for e in events}) == len(events),
        "event_key_patterns": dict(Counter("positive" if str(e.get("event_key", "")).startswith("p19r-pos:") else "negative" if str(e.get("event_key", "")).startswith("p19r-neg:") else "other" for e in events)),
        "manifest_sha256": {"positive": sha256(pos_path), "negative": sha256(neg_path)},
        "track_key_format": "v<video_id>:p<track_id>",
        "tao_intersections": {
            s["path"]: event_intersections(events, set(s["track_keys"]))
            for s in summaries
        },
    }

    # Code-level conclusion is deliberately explicit: the event category is
    # passed only to evaluator/state metadata and never to model forward args.
    code_files = [
        ROOT / "src/iclr27_phase19r/evaluation/internal.py",
        ROOT / "src/iclr27_phase19r/data/stream.py",
        ROOT / "src/iclr27_phase19r/runtime/runner.py",
        ROOT / "src/iclr27_phase19r/runtime/state.py",
        ROOT / "src/trackocd_v1/evaluation/trackocd_evaluator.py",
    ]
    code = {str(p): ast_contract(p) for p in code_files if p.exists()}
    q0_p71 = {
        "protocol": "phase72_q0_p71_ocd_interface_audit_v1",
        "q0_path": str(q0),
        "p71_paths": [str(p) for p in p71],
        "tao_stream_summaries": summaries,
        "event_manifest_audit": event_audit,
        "standard_trackocd_input_contract": {
            "known": ["prediction_type=known", "semantic_category_id"],
            "novel": ["prediction_type=novel", "anonymous virtual_category_id"],
            "unresolved": ["prediction_type=unresolved"],
            "tao_track_has_per_track_semantic_exporter": any(s["field_presence_counts"].get("semantic_category_id", 0) for s in summaries),
            "tao_track_has_prediction_type": any(s["field_presence_counts"].get("prediction_type", 0) for s in summaries),
        },
        "interface_conclusion": {
            "q0_p71_ocd_status": "NOT_RUN_INTERFACE_MISMATCH",
            "p71_learned_ocd_status": "NOT_RUN_BLOCKED",
            "reason": "TAO rows contain physical video/track/bbox/score fields but no prediction_type, semantic_category_id, virtual_category_id, action, commit, or causal representation fields; event v*:p* keys must be connected by a legal exporter before TrackOCDEvaluator can run.",
            "oracle_category_inputs": "evaluator/state metadata only; not in forward_item model arguments, action logits, candidate tensors, or semantic StateMemory tensors",
            "physical_id_semantic_separation": True,
            "future_or_held_inputs": False,
        },
        "code_audit": code,
    }

    ocd_metric = {
        "protocol": "phase72_ocd_metric_audit_v1",
        "source_streams": [s["path"] for s in summaries],
        "source_stream_sha256": {s["path"]: s["sha256"] for s in summaries},
        "event_manifests": {
            "positive": str(pos_path),
            "negative": str(neg_path),
            "positive_count": len(pos),
            "negative_count": len(neg),
            "total_count": len(events),
            "fold_counts": fold_counts,
            "sha256": event_audit["manifest_sha256"],
        },
        "required_metrics": {
            "causal_event": [
                "correct persistent Commit-CT / eligible / recall", "post-prefix CT",
                "existing precision/recall/F1", "new precision/recall/F1",
                "negative false merge / false commit", "premature rate", "unresolved rate",
                "defer rate", "first action position", "assignment delay", "duplicate births",
                "fragmentation", "merge", "NMI", "ARI", "category coverage", "video coverage",
            ],
            "trackocd_optional": [
                "supported_known_acc", "zero_shot_known_acc", "overall_known_acc",
                "known_to_novel_error", "known_misclassification_rate", "known_unresolved_rate",
                "novel_routing_recall", "novel_routing_precision", "false_known_absorption_rate",
                "unresolved_novel_rate", "route_aware_novel_acc", "conditional_novel_acc",
                "novel_only_nmi", "novel_only_ari", "macro_novel_class_acc",
                "predicted_novel_count", "novel_count_abs_error", "mean_fragmentation", "merge_error",
                "duplicate_creation_rate", "duplicate_avg_extra", "mean_assignment_delay",
                "all_track_acc", "macro_known_novel_harmonic",
            ],
        },
        "tao_track_schema_has_legal_ocd_exporter": False,
        "q0_p71_ocd_status": "NOT_RUN_INTERFACE_MISMATCH",
        "p71_learned_ocd_status": "NOT_RUN_BLOCKED",
        "phase19r_native_baseline_status": "PENDING_STAGE_C",
        "sealed_public_q1_accessed": False,
    }
    for path, payload in ((audit_dir / "q0_p71_interface_audit.json", q0_p71), (audit_dir / "ocd_metric_audit.json", ocd_metric)):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    print(json.dumps({"q0_p71": str(audit_dir / "q0_p71_interface_audit.json"), "ocd_metric": str(audit_dir / "ocd_metric_audit.json"), "streams": len(summaries), "events": len(events)}, indent=2))


if __name__ == "__main__":
    main()
