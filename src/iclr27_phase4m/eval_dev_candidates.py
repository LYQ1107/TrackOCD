"""Evaluate Phase 4M dev candidates: TrackEval + semantic + memory."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tag = args.tag
    dev = ROOT / "outputs" / "iclr27_phase4m" / "dev"
    te_root = dev / "trackeval"
    trackers_root = te_root / "trackers"
    trackers_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)

    subprocess.run([
        sys.executable,
        str(ROOT / "src/iclr27_phase4i/build_trackeval_input.py"),
        "--input-dir", str(dev / tag / "preds"),
        "--tracker-name", tag,
        "--output-root", str(trackers_root),
    ], check=True)
    subprocess.run([
        sys.executable,
        str(ROOT / "src/iclr27_phase4i/run_trackeval_subset.py"),
        "--trackers-folder", str(trackers_root),
        "--names", tag,
        "--out", str(te_root / f"tracking_{tag}.json"),
    ], env=env, check=True)
    env["PHASE4L_TAO_JSON"] = str(
        ROOT / "outputs/iclr27_phase3a/smoke/tao_subset/validation_20.json")
    subprocess.run([
        sys.executable,
        str(ROOT / "src/iclr27_phase4j/semantic_eval.py"),
        "--log-root", str(dev / tag / "semantic_logs"),
        "--out", str(te_root / f"semantic_{tag}.csv"),
        "--out-tracklets", str(te_root / f"tracklets_{tag}.csv"),
    ], env=env, check=True)
    env["PHASE4L_PROV_ROOT"] = str(
        ROOT / "outputs/iclr27_phase4m" / "audit")
    env["PHASE4L_AUDIT_OUT"] = str(
        ROOT / "outputs/iclr27_phase4m" / "audit")
    subprocess.run([
        sys.executable,
        str(ROOT / "src/iclr27_phase4k/build_offline_audit.py"),
        "--tag", tag,
    ], env=env, check=True)
    print("PHASE4M_EVAL_DEV_DONE", tag)


if __name__ == "__main__":
    main()
