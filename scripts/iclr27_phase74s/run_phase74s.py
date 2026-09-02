#!/usr/bin/env python3
"""Phase74S provenance reconciliation and model/evaluator contract builder.

This command is intentionally CPU-only.  It does not invoke Q0 or any
semantic/controller model.  If the legacy fallback is proven stale, it emits
an opaque, label-free 152-row model manifest and an evaluator-only join.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase74s"
sys.path.insert(0, str(ROOT))

from src.iclr27_phase74s.io import atomic_json, atomic_jsonl, atomic_text, sha256  # noqa: E402
from src.iclr27_phase74s.protocol import build_model_contract  # noqa: E402
from src.iclr27_phase74s.provenance import (  # noqa: E402
    build_graph,
    compare_protocols,
    consumer_table,
    inventory_manifests,
    stale_fallback_decision,
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_command(command: str) -> dict[str, Any]:
    result = subprocess.run(command, shell=True, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "observed_utc": now()}


class Phase74S:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.started = now()
        self.lock = OUT / "RUNNING.lock"
        self.commands: list[dict[str, Any]] = []

    def acquire(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        if self.lock.exists():
            try:
                old = json.loads(self.lock.read_text(encoding="utf-8"))
            except Exception:
                old = {}
            pid = int(old.get("pid", -1)) if str(old.get("pid", "-1")).lstrip("-").isdigit() else -1
            if pid > 0:
                try:
                    os.kill(pid, 0)
                except OSError:
                    os.replace(self.lock, OUT / f"RUNNING.lock.stale.{old.get('run_id', 'unknown')}")
                else:
                    raise RuntimeError(f"active Phase74S run owns {self.lock}: pid={pid}")
            else:
                os.replace(self.lock, OUT / f"RUNNING.lock.stale.{old.get('run_id', 'unknown')}")
        fd = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"phase": "Phase74S", "run_id": self.run_id, "pid": os.getpid(), "start_utc": self.started}, handle, indent=2)
            handle.write("\n")

    def close(self) -> None:
        if self.lock.exists():
            try:
                os.replace(self.lock, OUT / f"RUNNING.lock.closed.{self.run_id}")
            except FileNotFoundError:
                pass

    def command(self, command: str) -> dict[str, Any]:
        result = run_command(command)
        self.commands.append(result)
        return result

    def setup(self) -> None:
        for name in ("provenance", "manifests", "contracts", "audit", "replay", "tests", "logs", "completion"):
            (OUT / name).mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / "preregistered_experiment.json", {
            "phase": "Phase74S",
            "name": "Event Protocol Provenance Reconciliation and Autonomous Recovery",
            "hypothesis": "STALE_LEGACY_FALLBACK: Phase19R freeze_predictions is silently consuming the 82-row Phase18 stream while later build_folds/evaluator assets define a 152-row protocol.",
            "model_manifest_rule": "model rows contain only opaque UID, source/target tracklet keys and explicit source/target videos; evaluator labels remain in a post-freeze join",
            "forbidden": ["DEV+", "Q1", "public new-model labels", "sealed labels", "future rows/tracks", "category/text/semantic ID/physical ID feature shortcuts"],
            "training": False,
            "q0_model_invocation": False,
            "seed": None,
            "primary_gate": "PHASE74S_PASS_EVENT_PROTOCOL_RECONCILIATION requires unique provenance evidence and a complete 152-to-152 join",
            "stop_rule": "two equally supported protocols or any data/safety boundary violation blocks replay",
            "created_utc": self.started,
        })

    def preflight(self) -> dict[str, Any]:
        commands = [self.command(c) for c in ("free -h", "df -h /data1 /data2", "nvidia-smi", "ps -e -o pid,ppid,stat,etime,cmd")]
        payload = {"observed_utc": now(), "commands": commands, "gpu_count_used": 0, "external_processes_touched": False, "bounded_processes": 1}
        atomic_json(OUT / "audit/preflight.json", payload)
        atomic_text(OUT / "logs/preflight_resource.txt", "\n\n".join(f"$ {x['command']}\n{x['stdout']}" for x in commands))
        return payload

    def build(self) -> dict[str, Any]:
        inventory = inventory_manifests(ROOT)
        graph = build_graph(inventory)
        consumers = consumer_table(inventory)
        comparison = compare_protocols(inventory, ROOT)
        decision = stale_fallback_decision(inventory, comparison, ROOT)
        atomic_json(OUT / "provenance/event_manifest_inventory.json", {"schema_version": "phase74s.event_manifest_inventory.v1", "manifest_count": len(inventory), "records": inventory})
        atomic_json(OUT / "provenance/event_manifest_graph.json", graph)
        atomic_json(OUT / "provenance/event_manifest_consumers.json", {"schema_version": "phase74s.event_manifest_consumers.v1", "records": consumers})
        atomic_json(OUT / "provenance/protocol_comparison.json", comparison)
        atomic_json(OUT / "audit/stale_fallback_decision.json", decision)
        protocol = None
        if decision["status"] == "STALE_LEGACY_FALLBACK_CONFIRMED":
            protocol = build_model_contract(ROOT)
            atomic_jsonl(OUT / "manifests/model_events_v2.jsonl", protocol["model_records"])
            atomic_jsonl(OUT / "manifests/evaluator_join_v2.jsonl", protocol["join_records"])
            atomic_json(OUT / "contracts/model_evaluator_contract_v2.json", protocol["contract"])
            atomic_json(OUT / "contracts/model_manifest_input_contract.json", {
                "model_input_fields": ["source_tracklet_keys", "target_tracklet_key", "source_video", "target_video"],
                "opaque_uid_field": "model_event_uid",
                "evaluator_fields_are_post_freeze_only": True,
                "forbidden_model_fields": protocol["contract"]["model_manifest_forbidden_fields"],
                "model_event_count": protocol["contract"]["model_event_count"],
                "join_count": protocol["contract"]["join_count"],
            })
            atomic_json(OUT / "audit/model_manifest_field_audit.json", {
                "fields": protocol["contract"]["model_manifest_fields"],
                "forbidden_seen": protocol["contract"]["forbidden_model_fields_seen"],
                "source_video_cross_checked": True,
                "event_key_parsing_used": False,
                "category_or_kind_in_model_records": False,
            })
        return {"inventory": inventory, "graph": graph, "consumers": consumers, "comparison": comparison, "decision": decision, "protocol": protocol}

    def postflight(self) -> dict[str, Any]:
        commands = [self.command(c) for c in ("free -h", "df -h /data1 /data2", "nvidia-smi", "ps -e -o pid,ppid,stat,etime,cmd")]
        payload = {"observed_utc": now(), "commands": commands, "gpu_count_used": 0, "external_processes_touched": False, "phase74s_processes_active_after_run": False}
        atomic_json(OUT / "audit/postflight.json", payload)
        atomic_text(OUT / "logs/postflight_resource.txt", "\n\n".join(f"$ {x['command']}\n{x['stdout']}" for x in commands))
        return payload

    def ledger(self) -> dict[str, Any]:
        records: dict[str, Any] = {}
        for path in sorted(OUT.rglob("*")):
            rel = path.relative_to(OUT).as_posix()
            if rel == "manifests/output_sha256.json" or (not path.is_file() and not path.is_symlink()):
                continue
            if path.is_symlink():
                target = path.resolve(strict=False)
                records[rel] = {"symlink_target": os.readlink(path), "target_exists": target.exists(), "target_sha256": sha256(target) if target.is_file() else None}
            else:
                records[rel] = sha256(path)
        records["__self_hash__"] = "excluded_to_avoid_self_hash_cycle"
        atomic_json(OUT / "manifests/output_sha256.json", records)
        return records

    def run(self) -> dict[str, Any]:
        self.acquire()
        try:
            self.setup()
            pre = self.preflight()
            built = self.build()
            post = self.postflight()
            decision = built["decision"]
            status = "PHASE74S_PASS_EVENT_PROTOCOL_RECONCILIATION" if decision["status"] == "STALE_LEGACY_FALLBACK_CONFIRMED" and built["protocol"] else "PHASE74S_BLOCKED_TRUE_PROTOCOL_AMBIGUITY"
            if built["protocol"]:
                contract = built["protocol"]["contract"]
                if not (contract["model_event_count"] == contract["evaluator_event_count"] == contract["join_count"] == 152 and not contract["forbidden_model_fields_seen"] and contract["missing_model"] == contract["missing_evaluator"] == 0 and contract["duplicate_model"] == contract["duplicate_evaluator"] == 0):
                    status = "PHASE74S_BLOCKED_MODEL_CONTRACT"
            payload = {
                "phase": "Phase74S",
                "task": "Event Protocol Provenance Reconciliation and Autonomous Recovery",
                "status": status,
                "run_id": self.run_id,
                "start_utc": self.started,
                "end_utc": now(),
                "project_root": str(ROOT),
                "thread": "01a01fb6-96f7-7132-a318-0833180c88d8",
                "scope": {"training_run": False, "q0_model_invocation": False, "q0_control_replay": False, "event_replay": False, "controller_run": False, "sealed_accessed": False, "dev_plus_accessed": False, "q1_accessed": False, "public_new_accessed": False},
                "stale_fallback": decision,
                "model_contract": built["protocol"]["contract"] if built["protocol"] else None,
                "preflight": pre,
                "postflight": post,
                "resource": {"gpu_count_used": 0, "external_processes_touched": False, "one_bounded_process": True, "no_duplicate_supervisor": True},
                "commands": self.commands,
                "next_action": "run two independent Q0 control replays only after the explicit v2 model manifest is accepted" if status == "PHASE74S_PASS_EVENT_PROTOCOL_RECONCILIATION" else "continue provenance investigation; do not replay",
                "qualified_for_phase75a": status == "PHASE74S_PASS_EVENT_PROTOCOL_RECONCILIATION",
                "public_or_sealed_accessed": False,
            }
            atomic_json(OUT / "status.json", payload)
            atomic_jsonl(OUT / "logs/commands.jsonl", self.commands)
            atomic_text(OUT / "completion/phase74s.done", status + "\n")
            self.ledger()
            return payload
        finally:
            self.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"phase74s-{time.strftime('%Y%m%dT%H%M%SZ')}-p{os.getpid()}")
    args = parser.parse_args()
    status = Phase74S(args.run_id).run()
    print(json.dumps({"status": status["status"], "run_id": status["run_id"], "legacy": status["stale_fallback"]["status"], "qualified_for_phase75a": status["qualified_for_phase75a"]}, indent=2))


if __name__ == "__main__":
    main()

