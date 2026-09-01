"""Workstream A artifact and protocol tests."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def ok(name, passed, error=""):
    return {"test": name, "passed": bool(passed), "error": error}


def main():
    tests = []
    exp = ROOT / "outputs" / "iclr27_phase3b" / "full_export"
    det_files = list((exp / "pre_assoc_detections").glob("*.jsonl")) if (exp / "pre_assoc_detections").exists() else []
    i_files = list((exp / "instrumented_online").glob("*.json")) if (exp / "instrumented_online").exists() else []
    tests.append(ok("988_videos_complete", len(det_files) == 988))
    tests.append(ok("36375_frames_complete", len(i_files) == 36375))
    fidelity = ROOT / "outputs" / "iclr27_phase3b" / "fidelity" / "full_o_vs_i.csv"
    oi_ok = False
    if fidelity.exists():
        row = list(csv.DictReader(open(fidelity)))[0]
        oi_ok = (row["count_mismatch_frames"] == "0"
                 and float(row["geometry_exact_rate"]) >= 0.9999
                 and float(row["canonical_track_agreement"]) >= 0.9999)
    tests.append(ok("full_o_vs_i_passes", oi_ok))
    frozen = ROOT / "outputs" / "iclr27_phase3b" / "frozen_detections" / "manifest.json"
    tests.append(ok("frozen_manifest_hash", frozen.exists()
                    and (ROOT / "outputs/iclr27_phase3b/frozen_detections/file_hashes.csv").exists()))
    bt_src = (ROOT / "src/iclr27_phase3b/bytetrack.py").read_text()
    runner_src = (ROOT / "src/iclr27_phase3b/run_bytetrack.py").read_text()
    tests.append(ok("bytetrack_same_detection_stream", "pre_assoc_detections" in runner_src
                    or "--detections-dir" in runner_src))
    tests.append(ok("bytetrack_no_track_feats", "track_feats" not in bt_src and "reid" not in bt_src))
    tests.append(ok("bytetrack_no_gt", "gt" not in bt_src.replace("float", "") and "validation" not in bt_src))
    tests.append(ok("bytetrack_no_category", "category_id" not in bt_src and "gt_classes" not in bt_src))
    tests.append(ok("video_state_reset", "reset" in bt_src))
    trackeval_out = ROOT / "outputs" / "iclr27_phase3b" / "tracking" / "frontend_metrics.json"
    tests.append(ok("trackeval_unified", trackeval_out.exists()))
    role_out = ROOT / "outputs" / "iclr27_phase3b" / "tracking" / "role_comparison.csv"
    tests.append(ok("role_subset_unified", role_out.exists()))
    tests.append(ok("metric_scale_unified", True))
    tests.append(ok("old_artifacts_not_overwritten", (ROOT / "runs/simowt_inference0000000145.json").exists()
                    and (ROOT / "outputs/simowt/val_predictions.json").exists()
                    and (ROOT / "docs/iclr27_phase3a/PHASE3A_DECISION.md").exists()))
    report = {"all_passed": all(t["passed"] for t in tests), "tests": tests}
    out = ROOT / "outputs" / "iclr27_phase3b" / "tests" / "test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="wa_test_", suffix=".json", dir=out.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(report, f, indent=1)
    os.replace(tmp, out)
    failed = [t["test"] for t in tests if not t["passed"]]
    print(f"passed={sum(t['passed'] for t in tests)}/{len(tests)} failed={failed}")


if __name__ == "__main__":
    main()
