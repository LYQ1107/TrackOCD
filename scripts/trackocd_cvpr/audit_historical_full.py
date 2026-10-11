#!/usr/bin/env python3
"""One-worker retrospective audit; never invokes training or inference."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.historical_forensics import SealedForensics

SOURCES = (
    "configs/trackocd_cvpr/autonomous_research.json",
    "configs/trackocd_cvpr/cvpr0_historical_audit.json",
    "src/trackocd_cvpr/historical_forensics.py",
    "scripts/trackocd_cvpr/audit_historical_full.py",
)


def csv_write(path, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preregistration-commit", required=True)
    p.add_argument("--git-proxy", default="")
    args = p.parse_args()
    cfg = json.loads((ROOT / SOURCES[1]).read_text())
    safety = json.loads((ROOT / SOURCES[0]).read_text())
    git = ["git"] + (["-c", f"http.proxy={args.git_proxy}"] if args.git_proxy else []) + ["-c", "http.version=HTTP/1.1"]
    remote = subprocess.check_output(git + ["ls-remote", "origin", f"refs/heads/{safety['branch']}"],
                                     timeout=35, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}).split()[0]
    if remote != args.preregistration_commit:
        raise ValueError("Exact remote registration required before diagnostic")
    for name in SOURCES:
        if subprocess.check_output(["git", "show", f"{remote}:{name}"], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError("Diagnostic source differs from remote registration")
    out = ROOT / cfg["output_directory"]
    if out.exists():
        raise ValueError("Preserve actual diagnostic; no overwrite/retry")
    started = time.monotonic()

    def guard():
        mem = {k: int(v.split()[0]) * 1024 for k, v in
               (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
        if mem["MemAvailable"] < mem["MemTotal"] * safety["system_ram_reserve_fraction"]:
            raise RuntimeError("System RAM reserve")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg["rss_bytes_ceiling"]:
            raise RuntimeError("Planned process RSS ceiling")
        if time.monotonic() - started > cfg["wall_seconds_ceiling"]:
            raise RuntimeError("Wall ceiling")
        if shutil.disk_usage(ROOT).free < safety["disk_free_reserve_bytes"]:
            raise RuntimeError("Free disk reserve")

    guard()
    historical = ROOT / cfg["historical_root"]
    freeze_path = historical / "model_freeze.json"
    if sha256_file(freeze_path) != cfg["model_freeze_sha256"]:
        raise ValueError("Frozen historical models changed")
    freeze = json.loads(freeze_path.read_text())
    for name, digest in freeze["protected_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Historical protected input changed: {name}")
    manifest_path = historical / "limited_masa_evaluation/full_inference_manifest.json"
    public_path = ROOT / cfg["public_result"]
    public = json.loads(public_path.read_text())
    if sha256_file(manifest_path) != public["full_inference_manifest_sha256"]:
        raise ValueError("Historical manifest/result binding differs")
    manifest = json.loads(manifest_path.read_text())
    records = {r["job"]["id"]: r for r in manifest["records"]}
    cases = [c for c in public["cases"] if c["method"] == cfg["method"] and
             c["coverage_target"] == cfg["coverage_target"]]
    expected = {(s, o, c) for s in cfg["seeds"] for o in cfg["orders"] for c in cfg["prefixes"]}
    if len(cases) != cfg["expected_cases"] or {(c["seed"], c["order"], c["prefix"]) for c in cases} != expected:
        raise ValueError("Registered historical case universe differs")
    # Verify sealing and all selected ledgers BEFORE reading evaluator-only labels.
    for c in cases:
        r = records[c["execution_id"]]
        directory = historical / "limited_masa_evaluation/cases" / c["execution_id"]
        seal = json.loads((directory / "sealed_complete.json").read_text())
        if seal != r or r["status"] != "SEALED_COMPLETE_ALL_PREDICTED_IDS":
            raise ValueError("Incomplete or different sealed execution")
        ledger = directory / r["ledger"]["filename"]
        if ledger.stat().st_size != r["ledger"]["bytes"] or sha256_file(ledger) != r["ledger"]["sha256"]:
            raise ValueError("Historical decision ledger differs")
    join_path = ROOT / cfg["geometry_join"]
    if sha256_file(join_path) != cfg["geometry_join_sha256"]:
        raise ValueError("Frozen evaluator-only join differs")
    join = json.loads(join_path.read_text())
    if not join["evaluator_only_not_model_input"]:
        raise ValueError("Evaluator join isolation")
    targets = {}
    for row in join["frontends"]["MASA_NATIVE"]:
        if row["reliable_predicted_local_id"] is not None and row["role"] in {"known", "novel"}:
            key = (row["video_id"], int(row["reliable_predicted_local_id"]))
            if key in targets:
                raise ValueError("Non-unique geometry assignment")
            targets[key] = (row["role"], row["category_id"])
    roles = Counter(t[0] for t in targets.values())
    if roles != {"known": cfg["expected_reliable_known"], "novel": cfg["expected_reliable_novel"]}:
        raise ValueError("Reliable evaluator universe differs")
    import pyarrow.parquet as pq
    out.mkdir(parents=True)
    evidence, action_rows, error_rows, temporal_rows = [], [], [], []
    for c in sorted(cases, key=lambda c: (c["seed"], c["order"], c["prefix"])):
        guard()
        r = records[c["execution_id"]]
        ledger = historical / "limited_masa_evaluation/cases" / c["execution_id"] / r["ledger"]["filename"]
        analysis = SealedForensics(targets, cfg["expected_events_per_case"], cfg["temporal_bins"])
        for batch in pq.ParquetFile(ledger).iter_batches(batch_size=8192, use_threads=False,
                                                       columns=["video", "physical", "kind", "known", "token"]):
            arrays = [column.to_numpy() for column in batch.columns]
            for row in zip(*arrays):
                analysis.consume(*row)
            guard()
        a = analysis.finish()
        if a["counts"].get("correct_known", 0) != c["standard"]["old_correct"]:
            raise ValueError("Forensic Known count differs from sealed evaluation")
        if a["mixed_states"] != c["errors"]["contaminated_states"]:
            raise ValueError("Forensic mixed memory count differs")
        if a["counts"].get("proven_contamination_write_events", 0) != c["errors"]["memory_contamination_write_events"]:
            raise ValueError("Forensic contamination count differs")
        if a["counts"].get("unknown_anonymous_writes", 0) != c["errors"]["unmatched_anonymous_write_events"]:
            raise ValueError("Forensic unknown write count differs")
        key = {"seed": c["seed"], "order": c["order"], "prefix": c["prefix"]}
        a.update(key)
        a.update({"execution_id": c["execution_id"], "ledger_sha256": r["ledger"]["sha256"],
                  "old_correct": c["standard"]["old_correct"], "new_correct": c["standard"]["new_correct"],
                  "H": c["standard"]["h_score"], "correct_CT": c["persistent"]["commit_ct_correct"],
                  "reuse_coverage": c["persistent"]["effective_commit_coverage"],
                  "total_commitment_coverage": c["runtime"]["commitment_coverage"],
                  "wait_bias": r["job"]["wait_bias"]})
        evidence.append(a)
        for role in ("all", "known", "novel", "unknown"):
            for action in ("KNOWN", "NEW", "EXISTING", "WAIT"):
                action_rows.append({**key, "role": role, "action": action,
                                    "count": a["counts"].get(action if role == "all" else f"{role}_{action}", 0)})
        error_rows.append({**key, "first_bad_write_sequence": a["first_bad_write_sequence"],
                           "first_proven_contamination_sequence": a["first_proven_contamination_sequence"],
                           "mixed_states": a["mixed_states"], "unknown_states": a["states_with_unknown_members"],
                           "reuse_after_proven_contamination": a["counts"].get("reuse_after_proven_contamination", 0)})
        for b, counts in enumerate(a["temporal_bins"]):
            for action in ("KNOWN", "NEW", "EXISTING", "WAIT"):
                temporal_rows.append({**key, "temporal_bin": b, "action": action, "count": counts.get(action, 0)})
        atomic_json(out / "progress.json", {"status": "RUNNING_HISTORICAL_AUDIT", "completed": len(evidence), "total": len(cases)})
        print(json.dumps({"completed": len(evidence), **key}), flush=True)
    result = {
        "status": "COMPLETE_RETROSPECTIVE_SUBSTEP_NOT_COMPLETE_CVPR0",
        "preregistration_commit": remote,
        "config_sha256": sha256_file(ROOT / SOURCES[1]),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCES},
        "historical_inputs_sha256": {"model_freeze": sha256_file(freeze_path), "manifest": sha256_file(manifest_path),
                                     "public_metrics": sha256_file(public_path), "evaluator_join": sha256_file(join_path)},
        "protected_files_verified": len(freeze["protected_sha256"]),
        "cases": evidence,
        "raw_per_target_labels_or_predictions_published": False,
        "new_training_or_inference": False,
        "historical_val_model_selection": False,
        "TAO_Test_access": False,
        "resources": {"wall_seconds": time.monotonic() - started, "cpu_workers": 1, "GPU_hours": 0,
                      "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
        "pending": ["Train-development logits and top-K ranks", "checkpoint gradient replay",
                    "candidate/action oracle and first-error intervention", "representation geometry",
                    "same-population lawful predicted-track counterfactual"],
    }
    csv_write(out / "seed_action_distribution.csv", action_rows)
    csv_write(out / "memory_first_error.csv", error_rows)
    csv_write(out / "temporal_action_distribution.csv", temporal_rows)
    atomic_json(out / "result.json", result)
    atomic_json(out / "progress.json", {"status": result["status"], "completed": len(evidence), "total": len(cases)})
    print(json.dumps({"status": result["status"], "resources": result["resources"]}), flush=True)


if __name__ == "__main__":
    main()
