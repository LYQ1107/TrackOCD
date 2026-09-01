"""Re-evaluate Phase 4M held-out runs with the corrected GT."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TAO = ROOT / "outputs" / "iclr27_phase4n" / "audit" / \
    "validation_heldout_tao_corrected.json"
HO = ROOT / "outputs" / "iclr27_phase4n" / "heldout"
HO.mkdir(parents=True, exist_ok=True)
TAGS = ["j1b", "m1", "m3"]


def main():
    env = dict(os.environ)
    env["PHASE4L_TAO_JSON"] = str(TAO)
    env["PHASE4L_PROV_ROOT"] = str(
        ROOT / "outputs" / "iclr27_phase4m" / "audit" / "prov_ho_root")
    env["PHASE4L_AUDIT_OUT"] = str(ROOT / "outputs" / "iclr27_phase4n" /
                                   "audit" / "heldout_corrected")
    for tag in TAGS:
        log_root = ROOT / "outputs" / "iclr27_phase4m" / "runs" / "heldout" \
            / tag / "semantic_logs"
        subprocess.run([
            sys.executable,
            str(ROOT / "src" / "iclr27_phase4j" / "semantic_eval.py"),
            "--log-root", str(log_root),
            "--out", str(HO / f"semantic_{tag}.csv"),
            "--out-tracklets", str(HO / f"tracklets_{tag}.csv"),
        ], env=env, check=True)
        subprocess.run([
            sys.executable,
            str(ROOT / "src" / "iclr27_phase4m" /
                "build_identity_decisions_v2.py"),
            "--tag", tag,
            "--prov-root", str(ROOT / "outputs" / "iclr27_phase4m" /
                               "prov" / f"heldout_{tag}"),
            "--tao-json", str(TAO),
            "--z-cache", str(ROOT / "outputs" / "iclr27_phase4m" / "audit" /
                             "det_z_cache_heldout"),
            "--out", str(ROOT / "outputs" / "iclr27_phase4n" / "audit" /
                         f"identity_decisions_ho_{tag}_corrected.csv"),
        ], env=env, check=True)
        subprocess.run([
            sys.executable,
            str(ROOT / "src" / "iclr27_phase4k" / "build_offline_audit.py"),
            "--tag", tag,
        ], env=env, check=True)
        print("EVAL_HELDOUT_CORRECTED", tag, flush=True)
    print("HELDOUT_CORRECTED_DONE")


if __name__ == "__main__":
    main()
