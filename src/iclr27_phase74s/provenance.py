"""Event manifest provenance graph and protocol comparison helpers."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io import canonical_hash, iter_jsonl, sha256


EVENT_NAMES = (
    "positive_events.jsonl",
    "negative_events.jsonl",
    "identifiable_positive_events.jsonl",
    "identifiable_negative_events.jsonl",
    "held_known_positive_events.jsonl",
    "held_known_negative_events.jsonl",
    "held_known_model_events.jsonl",
    "public_model_events.jsonl",
    "model_events_v2.jsonl",
    "evaluator_join_v2.jsonl",
)
EVENT_PREFIXES = (
    "identifiable_positive_events",
    "identifiable_negative_events",
    "held_known_positive_events",
    "held_known_negative_events",
    "held_known_model_events",
    "public_model_events",
    "model_events_v2",
    "evaluator_join_v2",
)


def _manifest_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    # Only the requested project areas are searched.  This keeps the audit
    # bounded and avoids walking checkpoints/features or the .git object store.
    for area in ("data", "outputs", "src", "scripts", "configs", "docs"):
        base = root / area
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.name in EVENT_NAMES and (path.is_file() or path.is_symlink()):
                paths.append(path)
    return sorted(set(paths), key=lambda p: p.as_posix())


def _read_rows(path: Path) -> list[dict[str, Any]]:
    # All current event artifacts are JSONL.  Supporting a JSON array makes
    # the inventory explicit if a historical public manifest used that form.
    with path.open(encoding="utf-8") as handle:
        first = handle.read(1)
    if first == "[":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path} top-level JSON value is not a list")
        return [dict(x) for x in value]
    return list(iter_jsonl(path))


def _field_groups(fields: Iterable[str]) -> dict[str, list[str]]:
    names = sorted(set(fields))
    low = {name: name.lower() for name in names}
    category = [name for name in names if "category" in low[name] or low[name] in {"kind", "role", "polarity"}]
    gt = [name for name in names if any(token in low[name] for token in ("gt", "expected", "reliable", "target_row", "hard_negative"))]
    source = [name for name in names if low[name].startswith("source") or "source_track" in low[name]]
    target = [name for name in names if low[name].startswith("target") or "target_track" in low[name]]
    return {"category_fields": category, "GT_fields": gt, "source_track_fields": source, "target_track_fields": target}


def _phase_origin(path: Path) -> str:
    text = f"{path.as_posix()} {path.resolve(strict=False).as_posix()}"
    for phase in ("phase18", "phase19r", "phase72", "phase73", "phase74", "phase74r", "phase75"):
        if phase in text:
            return phase
    return "unknown"


def _protocol_name(path: Path) -> str:
    name = path.name
    if "identifiable" in name:
        return "Phase18_identifiable_82_event"
    if "held_known_model" in name:
        return "Phase19R_held_known_model_152_event"
    if "held_known_positive" in name or "held_known_negative" in name:
        return "Phase19R_held_known_evaluator_152_event"
    if "public_model" in name:
        return "Phase19R_public_model_event_stream"
    if "model_events_v2" in name:
        return "Phase74S_versioned_model_152_event"
    if "evaluator_join_v2" in name:
        return "Phase74S_evaluator_join_152_event"
    return "unknown_event_protocol"


def _code_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for area in ("src", "scripts", "configs"):
        base = root / area
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file() and path.suffix in {".py", ".sh", ".json", ".yaml", ".yml"})
    return sorted(files, key=lambda p: p.as_posix())


def _references(root: Path, path: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    tokens = {path.name, path.stem}
    for code in _code_files(root):
        try:
            lines = code.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        hits = [(idx + 1, line.strip()) for idx, line in enumerate(lines) if any(token in line for token in tokens)]
        if hits:
            roles: set[str] = set()
            text = "\n".join(line for _, line in hits).lower()
            if any(k in text for k in ("atomic_jsonl", "write_text", "os.replace", "build_", "generate")):
                roles.add("possible_generator")
            if any(k in text for k in ("read_text", "json.loads", "iter_jsonl", "open(", "load_jsonl")):
                roles.add("possible_consumer")
            refs.append({"path": str(code), "roles": sorted(roles) or ["reference"], "line_numbers": [n for n, _ in hits], "snippets": [s[:240] for _, s in hits[:8]]})
    return refs


def inventory_manifests(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in _manifest_paths(root):
        try:
            rows = _read_rows(path)
            parse_error = None
        except Exception as exc:  # preserve an explicit failure record
            rows = []
            parse_error = repr(exc)
        fields = sorted({field for row in rows for field in row})
        groups = _field_groups(fields)
        refs = _references(root, path)
        generator = [ref for ref in refs if "possible_generator" in ref["roles"]]
        consumers = [ref for ref in refs if "possible_consumer" in ref["roles"]]
        keys = [str(row.get("event_key", "")) for row in rows if "event_key" in row]
        record = {
            "path": str(path),
            "realpath": str(path.resolve(strict=False)),
            "is_symlink": path.is_symlink(),
            "sha256": sha256(path) if path.exists() and path.is_file() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
            "mtime_utc": __import__("datetime").datetime.fromtimestamp(path.stat().st_mtime, __import__("datetime").timezone.utc).isoformat() if path.exists() else None,
            "generator_script": [ref["path"] for ref in generator],
            "generator_function": "; ".join(f"{ref['path']}:{','.join(map(str, ref['line_numbers']))}" for ref in generator),
            "consumer_scripts": [ref["path"] for ref in consumers],
            "reference_evidence": refs,
            "row_count": len(rows),
            "event_key_count": len(keys),
            "event_key_unique": len(set(keys)) == len(keys),
            "event_key_hash": canonical_hash(keys),
            "fields": fields,
            **groups,
            "parse_error": parse_error,
            "phase_origin": _phase_origin(path),
            "protocol_name": _protocol_name(path),
        }
        inventory.append(record)
    return inventory


def build_graph(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for item in inventory:
        node_id = item["path"]
        nodes.append({"id": node_id, "type": "manifest", "protocol_name": item["protocol_name"], "phase_origin": item["phase_origin"], "row_count": item["row_count"], "sha256": item["sha256"]})
        for ref in item["reference_evidence"]:
            roles = ref["roles"]
            if "possible_generator" in roles:
                edges.append({"from": ref["path"], "to": node_id, "relation": "generates_or_writes", "line_numbers": ref["line_numbers"]})
            if "possible_consumer" in roles:
                edges.append({"from": node_id, "to": ref["path"], "relation": "consumed_or_read", "line_numbers": ref["line_numbers"]})
    return {"schema_version": "phase74s.event_manifest_graph.v1", "nodes": nodes, "edges": edges}


def consumer_table(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in inventory:
        out.append({"manifest": item["path"], "protocol_name": item["protocol_name"], "generator_script": item["generator_script"], "generator_function": item["generator_function"], "consumer_scripts": item["consumer_scripts"], "consumer_evidence": [{"path": x["path"], "line_numbers": x["line_numbers"], "snippets": x["snippets"]} for x in item["reference_evidence"] if "possible_consumer" in x["roles"]]})
    return out


def compare_protocols(inventory: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in inventory:
        groups.setdefault(item["protocol_name"], []).append(item)
    summary: dict[str, Any] = {}
    for protocol, items in sorted(groups.items()):
        unique_items: list[dict[str, Any]] = []
        seen_realpaths: set[str] = set()
        for item in items:
            if item["realpath"] in seen_realpaths:
                continue
            seen_realpaths.add(item["realpath"])
            unique_items.append(item)
        keys: list[str] = []
        for item in unique_items:
            try:
                keys.extend(str(row.get("event_key", "")) for row in _read_rows(Path(item["path"])))
            except Exception:
                pass
        summary[protocol] = {
            "manifest_paths": [x["path"] for x in unique_items],
            "realpaths": [x["realpath"] for x in unique_items],
            "counts": [x["row_count"] for x in unique_items],
            "total_rows": len(keys),
            "unique_event_keys": len(set(keys)),
            "event_key_hash": canonical_hash(keys),
            "fold_counts": dict(Counter()),
            "has_source_video": any("source_video" in x["fields"] for x in unique_items),
            "has_target_video": any("target_video" in x["fields"] for x in unique_items),
            "has_target_first_reliable_prefix": any("target_first_reliable_prefix_index_gt_only" in x["fields"] for x in unique_items),
            "category_fields": sorted({f for x in unique_items for f in x["category_fields"]}),
            "GT_fields": sorted({f for x in unique_items for f in x["GT_fields"]}),
        }
    # Current code/history evidence for the stale fallback hypothesis.
    freeze = root / "scripts/iclr27_phase19r/freeze_predictions.py"
    folds = root / "scripts/iclr27_phase19r/build_folds.py"
    internal = root / "src/iclr27_phase19r/evaluation/internal.py"
    freeze_text = freeze.read_text(encoding="utf-8") if freeze.exists() else ""
    folds_text = folds.read_text(encoding="utf-8") if folds.exists() else ""
    internal_text = internal.read_text(encoding="utf-8") if internal.exists() else ""
    summary["_code_history"] = {
        "freeze_predictions_exists": freeze.exists(),
        "freeze_has_silent_manifest_fallback": "if MANIFEST.exists()" in freeze_text and "data/iclr27_phase19r/sources" in freeze_text,
        "freeze_fallback_sorts_by_event_key": "rows.sort(key=lambda x: x[\"event_key\"])" in freeze_text,
        "build_folds_writes_held_model": "held_known_model_events.jsonl" in folds_text,
        "build_folds_writes_held_evaluator": "held_known_positive_events.jsonl" in folds_text and "held_known_negative_events.jsonl" in folds_text,
        "current_internal_evaluator_reads_held_manifests": "held_known_positive_events.jsonl" in internal_text and "held_known_negative_events.jsonl" in internal_text,
        "public_model_manifest_present": (root / "outputs/iclr27_phase19r/manifests/public_model_events.jsonl").exists(),
    }
    return summary


def stale_fallback_decision(inventory: list[dict[str, Any]], comparison: dict[str, Any], root: Path) -> dict[str, Any]:
    by_protocol = comparison
    def unique(protocol: str) -> list[dict[str, Any]]:
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for item in inventory:
            if item["protocol_name"] != protocol or item["realpath"] in seen:
                continue
            seen.add(item["realpath"])
            rows.append(item)
        return rows

    legacy_items = unique("Phase18_identifiable_82_event")
    legacy_counts = sorted(int(x["row_count"]) for x in legacy_items)
    eval_items = unique("Phase19R_held_known_evaluator_152_event")
    model_items = unique("Phase19R_held_known_model_152_event")
    code = comparison.get("_code_history", {})
    conditions = {
        "legacy_82_traceable_to_phase18": bool(legacy_counts and sum(legacy_counts) == 82 and all(x["phase_origin"] == "phase18" for x in legacy_items)),
        "later_152_evaluator_traceable_to_build_folds": bool(len(eval_items) >= 2 and all(x["row_count"] == 76 for x in eval_items) and code.get("build_folds_writes_held_evaluator")),
        "later_152_model_manifest_traceable_to_build_folds": bool(model_items and model_items[0]["row_count"] == 152 and code.get("build_folds_writes_held_model")),
        "phase72_or_internal_evaluator_uses_152": bool(code.get("current_internal_evaluator_reads_held_manifests") and eval_items and all(x["row_count"] == 76 for x in eval_items)),
        "152_hashes_are_frozen": bool(eval_items and all(len(str(x.get("sha256"))) == 64 for x in eval_items)),
        "82_is_not_current_evaluator_denominator": bool(legacy_counts and eval_items and sum(legacy_counts) != sum(int(x["row_count"]) for x in eval_items)),
        "freeze_fallback_reads_legacy_sources": bool(code.get("freeze_has_silent_manifest_fallback") and code.get("freeze_fallback_sorts_by_event_key")),
        "no_updated_82_evaluator_protocol": not any(x["row_count"] == 82 and x["protocol_name"] not in {"Phase18_identifiable_82_event"} for x in inventory),
    }
    confirmed = all(conditions.values())
    return {
        "schema_version": "phase74s.stale_fallback_decision.v1",
        "hypothesis": "STALE_LEGACY_FALLBACK",
        "status": "STALE_LEGACY_FALLBACK_CONFIRMED" if confirmed else "PHASE74S_BLOCKED_TRUE_PROTOCOL_AMBIGUITY",
        "conditions": conditions,
        "evidence": {
            "legacy_counts": legacy_counts,
            "evaluator_counts": [x["row_count"] for x in eval_items],
            "model_manifest_counts": [x["row_count"] for x in model_items],
            "public_model_manifest_present": code.get("public_model_manifest_present"),
            "protocol_comparison_keys": sorted(k for k in by_protocol if not k.startswith("_")),
        },
        "decision_rule": "all eight provenance conditions must be true; no positional/category mapping is allowed",
        "next_action": "generate versioned 152-row model manifest and evaluator-only join" if confirmed else "continue provenance search; do not replay",
    }
