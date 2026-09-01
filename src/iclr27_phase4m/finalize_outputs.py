"""Phase 4M final compliance outputs + report refresh."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
AUDIT = ROOT / "outputs" / "iclr27_phase4m" / "audit"
DEV = ROOT / "outputs" / "iclr27_phase4m" / "runs" / "dev"
HO = ROOT / "outputs" / "iclr27_phase4m" / "runs" / "heldout"
CSD = ROOT / "outputs" / "causal_semantic_deferral"


def concat(srcs, dst):
    rows = []
    for p in srcs:
        rows.extend(csv.DictReader(open(p)))
    if not rows:
        return
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    concat([AUDIT / f"identity_decisions_v2_{t}.csv"
            for t in ("j1b", "b1", "b2")],
           AUDIT / "identity_decisions.csv")
    concat([AUDIT / f"retrospective_{t}_k.csv"
            for t in ("j1b", "b1", "b2")],
           AUDIT / "time_to_resolution.csv")
    # method outputs under the task-required path
    for mode, src, tagmap in (
        ("dev", DEV, ["j1b", "m1", "m2", "m3"]),
        ("heldout", HO, ["j1b", "m1", "m3"]),
    ):
        out = CSD / mode
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "comparison.csv", out / "comparison.csv")
        rows = []
        for tag in tagmap:
            p = src / "trackeval" / f"resolution_{tag}.csv"
            if p.exists():
                rows.extend(csv.DictReader(open(p)))
        if rows:
            with open(out / "resolution_metrics.csv", "w",
                      newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        mrows = []
        for tag in tagmap:
            base = AUDIT / ("heldout" if mode == "heldout" else "")
            p = base / f"prototype_provenance_{tag}.csv"
            if p.exists():
                for r in csv.DictReader(open(p)):
                    mrows.append({"tag": tag, **r})
        if mrows:
            with open(out / "memory_quality.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(mrows[0].keys()))
                w.writeheader()
                w.writerows(mrows)
    print("FINALIZE_OUTPUTS_DONE")


if __name__ == "__main__":
    main()
