"""Evaluate Phase 4L dev candidates: TrackEval + semantic + memory."""
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
    dev = ROOT / "outputs" / "iclr27_phase4l" / "dev"
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
        ROOT / "outputs/iclr27_phase4l/audit")
    env["PHASE4L_AUDIT_OUT"] = str(
        ROOT / "outputs/iclr27_phase4l/audit")
    audit_link = ROOT / "outputs/iclr27_phase4l/audit" / f"prov_{tag}"
    prov_dev = ROOT / "outputs/iclr27_phase4l/audit" / f"prov_dev_{tag}"
    if not audit_link.exists() and prov_dev.exists():
        audit_link.symlink_to(prov_dev)
    subprocess.run([
        sys.executable,
        str(ROOT / "src/iclr27_phase4k/build_offline_audit.py"),
        "--tag", tag,
    ], env=env, check=True)
    print("EVAL_DEV_DONE", tag)


if __name__ == "__main__":
    main()
