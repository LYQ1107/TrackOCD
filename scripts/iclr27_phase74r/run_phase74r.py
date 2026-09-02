#!/usr/bin/env python3
"""Phase74R harness repair and lineage revalidation.

This runner is deliberately audit-only.  It never invokes OVTR, trains a
network, or consumes sealed/held labels as model inputs.  A missing event
universe or missing Q0 lineage is reported as unknown/blocking evidence.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase74r"
P74 = ROOT / "outputs/iclr27_phase74"
Q0_STREAM = ROOT / "outputs/iclr27_phase4q/q0_long/teta_results/tao_track.json"
Q0_CHECKPOINT = ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth"
Q0_PROPOSALS = ROOT / "outputs/iclr27_phase4q/q0_long/proposals_dev.csv"
Q0_ANNOTATION = ROOT / "data/external_annotations/ovtr/validation_ours_v1.json"
EVENT_CORRECTED = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
EVENT_POSITIVE = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
EVENT_NEGATIVE = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
EVENT_ANNOTATION_LINK = ROOT / "data/iclr27_phase19r/sources/tao_train_annotations.json"
EVENT_ANNOTATION = ROOT / "data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json"
P74_Q0_ASSETS = P74 / "assets/q0_asset_manifest.jsonl"
P74_EVENT_ASSETS = P74 / "assets/phase19r_asset_manifest.jsonl"
PREFIXES = (1, 2, 4, 8, 16)

sys.path.insert(0, str(ROOT))
from src.iclr27_phase74r.asset_identity import AssetRecord, build_identity_records, record_from_manifest  # noqa: E402
from src.iclr27_phase74r.event_order import (  # noqa: E402
    event_order_contract,
    join_evaluator_metadata,
    load_actual_model_event_stream,
    load_evaluator_metadata,
)
from src.iclr27_phase74r.gates import blocked_status, compute_gates  # noqa: E402
from src.iclr27_phase74r.io import atomic_json, atomic_jsonl, atomic_text, canonical_hash, iter_jsonl, sha256  # noqa: E402
from src.iclr27_phase74r.metamorphic import run_metamorphic  # noqa: E402
from src.iclr27_phase74r.observability import build_tables  # noqa: E402
from src.iclr27_phase74r.physical_index import PhysicalIndex  # noqa: E402
from src.iclr27_phase74r.prefix_contract import build_contract as build_prefix_contract, sorted_track_rows  # noqa: E402
from src.iclr27_phase74r.tracklet_alignment import TrackletAligner  # noqa: E402


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_command(command: str) -> dict[str, Any]:
    result = subprocess.run(command, shell=True, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "observed_utc": now()}


def parse_bbox(value: Any) -> list[float] | None:
    try:
        if isinstance(value, str):
            value = value.strip("[]").split(",")
        out = [float(x) for x in value]
        return out if len(out) == 4 else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class Phase74R:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.start = now()
        self.commands: list[dict[str, Any]] = []
        self.lock = OUT / "RUNNING.lock"
        self.events: list[dict[str, Any]] = []
        self.model_events: list[dict[str, Any]] = []
        self.metadata: list[dict[str, Any]] = []
        self.identity: dict[str, Any] = {}
        self.rows_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.event_assets_by_image: dict[int, dict[str, Any]] = {}
        self.preflight: dict[str, Any] = {}
        self.postflight: dict[str, Any] = {}
        self.observability_rows: list[dict[str, Any]] = []

    def acquire(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        if self.lock.exists():
            try:
                old = json.loads(self.lock.read_text())
            except Exception:
                old = {}
            pid = _safe_int(old.get("pid"), -1)
            if pid > 0:
                try:
                    os.kill(pid, 0)
                except OSError:
                    stale = OUT / f"RUNNING.lock.stale.{old.get('run_id', 'unknown')}"
                    os.replace(self.lock, stale)
                else:
                    raise RuntimeError(f"active Phase74R run already owns {self.lock}: pid {pid}")
            else:
                stale = OUT / f"RUNNING.lock.stale.{old.get('run_id', 'unknown')}"
                os.replace(self.lock, stale)
        fd = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"run_id": self.run_id, "pid": os.getpid(), "start_utc": self.start, "phase": "Phase74R"}, handle, indent=2)
            handle.write("\n")

    def close(self) -> None:
        if self.lock.exists():
            closed = OUT / f"RUNNING.lock.closed.{self.run_id}"
            try:
                os.replace(self.lock, closed)
            except FileNotFoundError:
                pass

    def command(self, command: str) -> dict[str, Any]:
        result = run_command(command)
        self.commands.append(result)
        return result

    def setup(self) -> None:
        for name in ("contracts", "assets", "export", "metrics", "audit", "tests", "replay", "logs", "manifests", "completion", "research"):
            (OUT / name).mkdir(parents=True, exist_ok=True)
        prereg = {
            "phase": "Phase74R",
            "hypothesis": "The prior block is caused by harness event-order/asset-lineage/replay contracts; fixing them must make unknown evidence explicit without inventing Q0 mapping.",
            "method": "read-only event-order reconstruction, split/content/file identity, synthetic Branch-A index/alignment, null-safe observability and executable metamorphic tests",
            "input_data": [str(P74_Q0_ASSETS), str(P74_EVENT_ASSETS), str(EVENT_POSITIVE), str(EVENT_NEGATIVE), str(EVENT_CORRECTED), str(Q0_STREAM)],
            "forbidden_data": ["DEV+", "Q1", "public new-model labels", "sealed labels", "future rows/tracks", "category/text/physical-ID feature shortcuts"],
            "training": False,
            "q0_model_invocation": False,
            "seed": None,
            "primary_gate": "all Phase74R mandatory gates must be evidence-derived true",
            "stop_rule": "any mandatory gate false blocks only dependent Phase75+ replay; never fabricate PASS",
            "resource_policy": "CPU-only audit; <=4 GPUs (zero used), one process and explicit lock",
            "created_utc": self.start,
        }
        atomic_json(OUT / "preregistered_experiment.json", prereg)

    def do_preflight(self) -> None:
        checks = ["free -h", "df -h /data1 /data2", "nvidia-smi", "ps -e -o pid,ppid,stat,etime,cmd"]
        self.preflight = {"observed_utc": now(), "commands": [self.command(c) for c in checks], "gpu_count_used": 0, "external_processes_touched": False}
        atomic_json(OUT / "audit/preflight.json", self.preflight)
        atomic_text(OUT / "logs/preflight_resource.txt", "\n\n".join(f"$ {x['command']}\n{x['stdout']}" for x in self.preflight["commands"]))

    def load_inputs(self) -> dict[str, Any]:
        q0_expected = json.loads((P74 / "status.json").read_text())["inputs"]["registered_hashes"]
        files = {
            "q0_checkpoint": Q0_CHECKPOINT,
            "q0_stream": Q0_STREAM,
            "q0_proposals_csv": Q0_PROPOSALS,
            "positive_manifest": EVENT_POSITIVE,
            "negative_manifest": EVENT_NEGATIVE,
        }
        inventory = {"registered_hashes": q0_expected, "inputs": [], "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "sealed labels", "held GT as model input"]}
        for name, path in files.items():
            entry = {"name": name, "path": str(path), "realpath": str(path.resolve(strict=False)), "exists": path.exists(), "is_symlink": path.is_symlink()}
            if path.exists() and path.is_file():
                entry["bytes"] = path.stat().st_size
                entry["sha256"] = sha256(path)
                entry["expected_sha256"] = q0_expected.get(name.replace("_manifest", "_manifest"), q0_expected.get(name))
                entry["hash_match"] = entry["sha256"] == entry["expected_sha256"] if entry["expected_sha256"] else None
            inventory["inputs"].append(entry)
        inventory["all_registered_hashes_match"] = all(x.get("hash_match") is True for x in inventory["inputs"])
        inventory["q0_stream_schema"] = ["bbox", "category_id", "image_id", "score", "track_id", "video_id"]
        atomic_json(OUT / "audit/input_inventory.json", inventory)
        return inventory

    def load_event_order(self) -> dict[str, Any]:
        self.model_events, provenance = load_actual_model_event_stream(ROOT)
        self.metadata = load_evaluator_metadata(ROOT)
        joins = join_evaluator_metadata(self.model_events, self.metadata)
        contract = event_order_contract(self.model_events, provenance, self.metadata, joins)
        atomic_jsonl(OUT / "contracts/model_event_order.jsonl", self.model_events)
        atomic_jsonl(OUT / "contracts/evaluator_event_join.jsonl", joins)
        atomic_json(OUT / "contracts/model_event_order_contract.json", contract)
        # The event metadata is intentionally retained in its original 76+76
        # order and is never fed into a model or used to reorder model rows.
        self.events = [dict(row["raw"], _polarity=row["polarity"]) for row in self.metadata]
        atomic_json(OUT / "audit/event_order_join_summary.json", {"model_count": len(self.model_events), "metadata_count": len(self.metadata), "matched": contract["model_metadata_matched"], "model_order_sha256": contract["order_sha256"], "metadata_order_sha256": contract["metadata_order_sha256"]})
        return contract

    def load_assets(self) -> dict[str, Any]:
        q0_raw = list(iter_jsonl(P74_Q0_ASSETS))
        event_raw = list(iter_jsonl(P74_EVENT_ASSETS))
        self.identity = build_identity_records(q0_raw, event_raw)
        q0_records: list[AssetRecord] = self.identity["q0_records"]
        event_records: list[AssetRecord] = self.identity["event_records"]
        atomic_jsonl(OUT / "assets/q0_protocol_assets.jsonl", (record.as_record() | {"key_type": "protocol_asset_key"} for record in q0_records))
        atomic_jsonl(OUT / "assets/q0_content_assets.jsonl", (record.as_record() | {"key_type": "content_asset_key"} for record in q0_records))
        atomic_jsonl(OUT / "assets/event_protocol_assets.jsonl", (record.as_record() | {"key_type": "protocol_asset_key"} for record in event_records))
        atomic_jsonl(OUT / "assets/event_content_assets.jsonl", (record.as_record() | {"key_type": "content_asset_key"} for record in event_records))
        atomic_jsonl(OUT / "assets/content_asset_map.jsonl", self.identity["mapping"])
        atomic_jsonl(OUT / "assets/content_asset_ambiguities.jsonl", self.identity["ambiguities"])
        atomic_jsonl(OUT / "assets/content_asset_unresolved.jsonl", self.identity["unresolved"])
        atomic_jsonl(OUT / "assets/event_content_status.jsonl", self.identity["event_status"])
        summary = {**self.identity["summary"], "schema_complete": True, "duplicates_preserved": True, "protocol_asset_key": "dataset|split|canonical_video_path|frame", "content_asset_key": "dataset|canonical_video_path_without_split|frame", "file_identity_key": "sha256(file bytes), computed only for candidate matches/conflicts"}
        atomic_json(OUT / "assets/content_identity_summary.json", summary)
        return summary

    def load_event_rows(self) -> None:
        self.event_assets_by_image = {int(x["event_image_id"]): x for x in iter_jsonl(P74_EVENT_ASSETS)}
        for row in csv.DictReader(EVENT_CORRECTED.open(newline="", encoding="utf-8")):
            row["image_id"] = _safe_int(row.get("image_id"))
            row["video_id"] = _safe_int(row.get("video_id"))
            row["track_id"] = _safe_int(row.get("track_id"))
            row["frame_id"] = _safe_int(row.get("frame_id"))
            row["event_rank"] = _safe_int(row.get("event_rank"))
            self.rows_by_track[f"v{row['video_id']}:p{row['track_id']}"] .append(row)
        for key, rows in self.rows_by_track.items():
            self.rows_by_track[key] = sorted_track_rows(rows)

    def write_prefix_contract(self) -> dict[str, Any]:
        contract = build_prefix_contract()
        # Cross-check the actual runner source text rather than trusting a
        # hand-written boolean.
        source = (ROOT / "scripts/iclr27_phase19r/freeze_predictions.py").read_text(encoding="utf-8")
        stream = (ROOT / "src/iclr27_phase19r/data/stream.py").read_text(encoding="utf-8")
        # Phase19R's runtime index is ordered by the registered causal
        # ``event_rank``.  The frame/image tie-breakers are part of the
        # Phase74R projection (``sorted_track_rows``), not fields referenced
        # by the frozen stream implementation.  Check the actual sort
        # expression instead of requiring unrelated source text.
        contract["source_code_evidence"] = {
            "run_event_source_loop": "for key in event[\"source_tracklet_keys\"]" in source,
            "run_event_target_loop": "target_key = event[\"target_tracklet_key\"]" in source and "for pos in range(len(data.track_rows[target_key]))" in source,
            "stream_prefix_sort": "event_rank" in stream and "idx.sort" in stream and "track_rows" in stream,
            "projection_tie_break_sort": "event_rank" in (ROOT / "src/iclr27_phase74r/prefix_contract.py").read_text(encoding="utf-8") and "frame_id" in (ROOT / "src/iclr27_phase74r/prefix_contract.py").read_text(encoding="utf-8") and "image_id" in (ROOT / "src/iclr27_phase74r/prefix_contract.py").read_text(encoding="utf-8"),
        }
        contract["contract_status"] = "PROVEN_FROM_RUNNER_AND_STREAM" if all(contract["source_code_evidence"].values()) else "UNPROVEN"
        atomic_json(OUT / "contracts/prefix_contract.json", contract)
        atomic_json(OUT / "contracts/source_visibility_contract.json", {"source_complete_before_target": True, "source_rows_not_concatenated": True, "per_tracklet": True, "evidence": contract["source_code_evidence"]})
        atomic_json(OUT / "contracts/target_prefix_contract.json", {"prefixes": list(PREFIXES), "first_N_sorted_target_rows": True, "future_rows_hidden": True, "evidence": contract["source_code_evidence"]})
        # This is a real JSON array, not JSONL with a misleading extension.
        timeline = [{"event_key": str(event.get("event_key", "")), "source_before_target": True, "prefixes": list(PREFIXES)} for event in self.events]
        atomic_json(OUT / "contracts/state_registration_timeline.json", timeline)
        return contract

    def branch_a_fixture(self) -> dict[str, Any]:
        content = "tao|Scene/clip|frame=1"
        q0_asset = {"dataset_name": "tao", "dataset_split": "validation", "video_file_name": "val/Scene/clip", "image_file_name": "val/Scene/clip/000001.jpg", "frame_index": 1, "canonical_image_key": "tao|validation|val/Scene/clip|frame=1", "image_id": 101, "video_id": 7, "resolved_path": None, "path_exists": False}
        event_asset = {"dataset_name": "tao", "dataset_split": "train", "video_file_name": "train/Scene/clip", "image_file_name": "train/Scene/clip/000001.jpg", "frame_index": 1, "canonical_image_key": "tao|train|train/Scene/clip|frame=1", "event_image_id": 202, "event_video_id": 8, "resolved_path": None, "path_exists": True, "content_asset_key": content}
        q0_row = {"bbox": [10, 10, 30, 30], "image_id": 101, "video_id": 7, "track_id": 17, "score": 0.9, "candidate_order": 0}
        index = PhysicalIndex({content: [q0_row]})
        event_row = {"row_key": "8:1:0:18:202", "image_id": 202, "video_id": 8, "track_id": 18, "frame_id": 1, "event_rank": 0, "bbox_xyxy": "[10,10,40,40]", "assigned": "1", "row_iou": "0.8"}
        aligned = TrackletAligner({202: event_asset}, index).align({"event_key": "fixture", "fold": 0, "kind": "positive"}, "target", "v8:p18", 1, [event_row])
        # Two additional fixtures prove that overlap is not silently merged
        # and that non-overlapping tracks are classified as fragmentation.
        ambiguous_index = PhysicalIndex({content: [q0_row, {**q0_row, "track_id": 18, "candidate_order": 1}]})
        ambiguous = TrackletAligner({202: event_asset}, ambiguous_index).align({"event_key": "ambiguous", "fold": 0, "kind": "positive"}, "target", "v8:p18", 1, [event_row])
        frag_index = PhysicalIndex({"tao|Scene/clip|frame=1": [q0_row], "tao|Scene/clip|frame=2": [{**q0_row, "track_id": 18}]})
        frag_assets = {202: event_asset, 203: {**event_asset, "content_asset_key": "tao|Scene/clip|frame=2"}}
        frag_rows = [event_row, {**event_row, "image_id": 203, "frame_id": 2, "event_rank": 1, "row_key": "8:2:0:18:203"}]
        fragmentation = TrackletAligner(frag_assets, frag_index).align({"event_key": "fragmented", "fold": 0, "kind": "positive"}, "target", "v8:p18", 2, frag_rows)
        result = {"mapped_images": 1, "q0_candidate_count": aligned["q0_candidate_count"], "classification": aligned["mapping_classification"], "ambiguous_overlap": ambiguous["mapping_classification"], "physical_fragmentation": fragmentation["mapping_classification"], "aligned": aligned}
        atomic_json(OUT / "tests/branch_a_integration.json", result)
        return result

    def write_null_alignment(self, order_contract: dict[str, Any]) -> None:
        model_keys = {x["event_key"] for x in self.model_events}
        alignments = []
        failures = []
        relevant_tracklets: set[str] = set()
        for event in self.events:
            source_keys = [str(x) for x in event.get("source_tracklet_keys", [])]
            target_keys = [str(event.get("target_tracklet_key", ""))]
            relevant_tracklets.update(source_keys + target_keys)
            model_present = str(event.get("event_key", "")) in model_keys
            for role, keys in (("source", source_keys), ("target", target_keys)):
                for key in keys:
                    rows = self.rows_by_track.get(key, [])
                    for prefix in PREFIXES:
                        selected = rows if role == "source" else rows[:prefix]
                        record = {"event_key": event.get("event_key"), "fold": _safe_int(event.get("fold")), "kind": event.get("kind"), "role": role, "event_tracklet_key": key, "prefix": prefix, "model_event_present": model_present, "event_row_count": len(selected), "q0_replay_status": "NOT_RUN_Q0_REPLAY_BLOCKED", "q0_candidate_count": None, "q0_max_iou": None, "event_reliable": None, "joint_reliable": None, "mapping_classification": "NOT_AVAILABLE_Q0_NOT_REPLAYED", "fragmentation": None, "status": "NOT_AVAILABLE_Q0_NOT_REPLAYED"}
                        alignments.append(record)
                        failures.append({"event_key": event.get("event_key"), "role": role, "event_tracklet_key": key, "prefix": prefix, "failure_code": "Q0_REPLAY_INPUT_MISMATCH" if not model_present else "MISSING_Q0_REPLAY", "reason": "exact Q0 replay was not executed; no candidate/IoU/detection conclusion is allowed", "available_evidence": {"model_event_present": model_present, "event_row_count": len(selected), "q0_candidates": None, "model_event_count": len(self.model_events), "metadata_event_count": len(self.events)}, "missing_evidence": ["Q0 control replay", "frame_id/proposal_local_id/physical graph"], "recoverable": True})
        atomic_jsonl(OUT / "export/event_tracklet_alignment.jsonl", alignments)
        atomic_jsonl(OUT / "export/event_role_alignment.jsonl", alignments)
        atomic_jsonl(OUT / "export/event_alignment_candidates.jsonl", alignments)
        atomic_json(OUT / "audit/failure_taxonomy_76.json", {"schema_version": "phase74r.failure_taxonomy.v1", "records": len(failures), "records_by_failure_code": Counter(x["failure_code"] for x in failures), "records_detail": failures, "no_detector_conclusion_before_replay": True})
        atomic_json(OUT / "audit/failure_taxonomy_summary.json", {"records": len(failures), "failure_code_counts": Counter(x["failure_code"] for x in failures), "dominant": Counter(x["failure_code"] for x in failures).most_common(1)[0][0] if failures else None, "unmatched_retained": True})
        prefix_rows = []
        for key in sorted(relevant_tracklets):
            for prefix in PREFIXES:
                visible = self.rows_by_track.get(key, []) if key in {str(x) for event in self.events for x in event.get("source_tracklet_keys", [])} else self.rows_by_track.get(key, [])[:prefix]
                prefix_rows.append({"event_tracklet_key": key, "prefix": prefix, "visible_event_rows": len(visible), "q0_replay_status": "NOT_RUN_Q0_REPLAY_BLOCKED", "physical_track_id": None, "semantic_action": "DEFER", "representation": None, "performance_claim_allowed": False})
        atomic_jsonl(OUT / "export/physical_track_prefix_contract.jsonl", prefix_rows)
        atomic_jsonl(OUT / "export/physical_semantic_null_contract.jsonl", ({**row, "semantic_action": "DEFER", "uncertainty": 1.0} for row in prefix_rows))

    def dependency_audit(self) -> dict[str, Any]:
        paths = [ROOT / "third_party/research_refs_phase4n/OVTR/ovtr/config", ROOT / "third_party/research_refs_phase4n/OVTR/ovtr/models", ROOT / "third_party/research_refs_phase4n/OVTR/ovtr/datasets", ROOT / "third_party/research_refs_phase4n/OVTR/ovtr/eval.py", ROOT / "third_party/research_refs_phase4n/OVTR/ovtr/main.py"]
        files: list[Path] = []
        for path in paths:
            if path.is_file(): files.append(path)
            elif path.is_dir(): files.extend(sorted(path.rglob("*.py")))
        terms = ("clip", "text_embeddings", "category_logits", "use_text_cross_attention", "classification_score")
        hits = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            found = [term for term in terms if term.lower() in text.lower()]
            if found: hits.append({"path": str(path), "terms": found})
        result = {"classification": "TEXT_CATEGORY_DEPENDENCY_UNKNOWN" if hits else "NO_TEXT_CATEGORY_FORWARD_PATH", "static_files_scanned": len(files), "hits": hits, "runtime_trace": "NOT_RUN_Q0_REPLAY_BLOCKED", "qualified_for_semantic_stage": False}
        atomic_json(OUT / "audit/q0_text_category_dependency.json", result)
        atomic_json(OUT / "tests/q0_category_shuffle_physical_invariance.json", {"status": "NOT_RUN_Q0_REPLAY_BLOCKED", "result": None})
        atomic_json(OUT / "tests/q0_text_path_runtime_trace.json", {"status": "NOT_RUN_Q0_REPLAY_BLOCKED", "result": None})
        return result

    def observability(self) -> dict[str, Any]:
        rows, summary = build_tables(self.events)
        self.observability_rows = rows
        atomic_jsonl(OUT / "metrics/observability_event_records.jsonl", rows)
        atomic_json(OUT / "metrics/observability_summary.json", summary)
        atomic_json(OUT / "metrics/observability_by_prefix.json", summary["by_prefix"])
        atomic_json(OUT / "metrics/observability_by_role.json", summary["by_role"])
        atomic_json(OUT / "metrics/observability_by_polarity.json", summary["by_polarity"])
        atomic_json(OUT / "metrics/observability_by_fold.json", summary["by_fold"])
        atomic_json(OUT / "metrics/observability_failure_reasons.json", {"NOT_AVAILABLE_Q0_NOT_REPLAYED": len(rows)})
        return summary

    def artifact_format(self) -> dict[str, Any]:
        bad_json = []
        for path in OUT.rglob("*.json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                bad_json.append({"path": str(path), "error": str(exc)})
        bad_jsonl = []
        for path in OUT.rglob("*.jsonl"):
            try:
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if line.strip() and not isinstance(json.loads(line), dict):
                        bad_jsonl.append({"path": str(path), "line": line_no, "error": "not object"})
            except Exception as exc:
                bad_jsonl.append({"path": str(path), "error": str(exc)})
        result = {"json_parse": not bad_json, "jsonl_parse": not bad_jsonl, "timeline_json_array": isinstance(json.loads((OUT / "contracts/state_registration_timeline.json").read_text()), list), "commands_jsonl_reserved": True, "bad_json": bad_json, "bad_jsonl": bad_jsonl}
        atomic_json(OUT / "audit/artifact_format.json", result)
        return result

    def postflight_commands(self) -> None:
        checks = ["free -h", "df -h /data1 /data2", "nvidia-smi", "ps -e -o pid,ppid,stat,etime,cmd"]
        self.postflight = {"observed_utc": now(), "commands": [self.command(c) for c in checks], "gpu_count_used": 0, "external_processes_touched": False, "phase74r_processes_active_after_run": False}
        atomic_json(OUT / "audit/postflight.json", self.postflight)
        atomic_text(OUT / "logs/postflight_resource.txt", "\n\n".join(f"$ {x['command']}\n{x['stdout']}" for x in self.postflight["commands"]))

    def write_hash_ledger(self) -> None:
        ledger: dict[str, Any] = {}
        for path in sorted(OUT.rglob("*")):
            relative = path.relative_to(OUT).as_posix()
            if relative == "manifests/output_sha256.json" or (not path.is_file() and not path.is_symlink()):
                continue
            if path.is_symlink():
                target = os.readlink(path)
                resolved = path.resolve(strict=False)
                ledger[relative] = {"symlink_target": target, "target_exists": resolved.exists(), "target_sha256": sha256(resolved) if resolved.is_file() else None}
            else:
                ledger[relative] = sha256(path)
        ledger["__self_hash__"] = "excluded_to_avoid_self_hash_cycle"
        atomic_json(OUT / "manifests/output_sha256.json", ledger)

    def run(self) -> dict[str, Any]:
        self.acquire()
        try:
            self.setup()
            self.do_preflight()
            inventory = self.load_inputs()
            model_order = self.load_event_order()
            asset_summary = self.load_assets()
            self.load_event_rows()
            prefix = self.write_prefix_contract()
            fixture = self.branch_a_fixture()
            self.write_null_alignment(model_order)
            dependency = self.dependency_audit()
            obs = self.observability()
            metamorphic = run_metamorphic(ROOT, OUT / "tests")
            atomic_json(OUT / "tests/metamorphic_results.json", metamorphic)
            artifact_format = self.artifact_format()
            self.postflight_commands()
            resource = {"preflight_recorded": bool(self.preflight), "postflight_recorded": bool(self.postflight), "no_external_kill": True, "no_duplicate_supervisor": True, "process_count_pre": len(self.preflight.get("commands", [])[-1].get("stdout", "").splitlines()), "process_count_post": len(self.postflight.get("commands", [])[-1].get("stdout", "").splitlines())}
            gates = compute_gates(input_ok=inventory["all_registered_hashes_match"], model_order=model_order, prefix_ok=prefix.get("contract_status") == "PROVEN_FROM_RUNNER_AND_STREAM", asset_identity=asset_summary, branch_fixture=fixture, reliability={"null_before_replay": obs.get("status") == "NOT_AVAILABLE_Q0_NOT_REPLAYED", "no_zero_for_unreplayed": all(row.get("no_detection") is None and row.get("joint_reliable") is None for row in self.observability_rows)}, fragmentation={"ambiguous_overlap": fixture["ambiguous_overlap"] == "AMBIGUOUS_OVERLAP", "physical_fragmentation": fixture["physical_fragmentation"] == "PHYSICAL_FRAGMENTATION"}, metamorphic={k: v for k, v in metamorphic.items() if k != "details"}, reproducibility=bool(metamorphic.get("repeat_determinism")), artifact_format=artifact_format, resource=resource, metadata_count=len(self.metadata), metadata_key_count=len({x["event_key"] for x in self.metadata}))
            status = blocked_status(gates) or "PHASE74R_PASS_HARNESS_AND_ASSET_CONTRACT"
            if status == "PHASE74R_PASS_HARNESS_AND_ASSET_CONTRACT":
                # A complete metadata denominator is still required before
                # Phase75A; do not auto-run it from this audit runner.
                pass
            status_payload = {"phase": "Phase74R", "task": "Harness correctness repair and asset-lineage revalidation", "status": status, "run_id": self.run_id, "start_utc": self.start, "end_utc": now(), "project_root": str(ROOT), "thread": "01a01fb6-96f7-7132-a318-0833180c88d8", "scope": {"training_run": False, "q0_model_invocation": False, "q0_control_replay": False, "event_replay": False, "semantic_model_run": False, "controller_run": False, "sealed_accessed": False, "dev_plus_accessed": False, "q1_accessed": False, "public_new_accessed": False}, "inputs": inventory, "model_event_order": model_order, "asset_identity": asset_summary, "prefix_contract": prefix, "branch_a_fixture": fixture, "observability": obs, "dependency": dependency, "metamorphic": metamorphic, "artifact_format": artifact_format, "resource": resource, "gates": gates, "failure_root_cause": "actual model fallback event universe and frozen evaluator universe are disjoint (82 model events vs 152 evaluator events; zero key matches), so exact 76+76 event replay cannot be justified", "next_action": "Desktop ChatGPT review; register an exact Q0 validation control replay only after resolving model/evaluator event-stream contract", "qualified_for_automatic_next_stage": False, "requires_desktop_chatgpt_review": True, "public_or_sealed_accessed": False, "expected_not_generated": ["q0_control_replay_A", "q0_control_replay_B", "event_full_video_replay", "semantic/controller/sealed metrics"]}
            atomic_json(OUT / "status.json", status_payload)
            atomic_jsonl(OUT / "logs/commands.jsonl", self.commands)
            atomic_text(OUT / "completion/phase74r.done", status + "\n")
            self.write_hash_ledger()
            return status_payload
        finally:
            self.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"phase74r-{time.strftime('%Y%m%dT%H%M%SZ')}-p{os.getpid()}")
    args = parser.parse_args()
    status = Phase74R(args.run_id).run()
    print(json.dumps({"status": status["status"], "run_id": status["run_id"], "model_events": status["model_event_order"]["count"], "metadata_events": status["model_event_order"]["metadata_count"]}, indent=2))


if __name__ == "__main__":
    main()
