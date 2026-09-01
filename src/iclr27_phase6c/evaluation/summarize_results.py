"""Collect Phase 6C eval JSONs into a single comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


STRICT_KEYS = [
    "known_occurrence_acc", "first_novel_birth_acc", "novel_reuse_acc",
    "cross_physical_reuse_acc", "cross_physical_reuse_share",
    "known_to_new_rate", "known_to_existing_rate",
    "n_born_novel_states", "n_true_novel_categories", "novel_count_abs_error",
    "mean_fragmentation", "duplicate_creation_rate",
    "novel_nmi", "novel_ari", "semantic_switch_rate",
]
LEGACY_KEYS = [
    "overall_known_acc", "route_aware_novel_acc", "conditional_novel_acc",
    "novel_only_nmi", "novel_only_ari", "all_track_acc",
    "predicted_novel_count", "novel_count_abs_error",
]


def load_run(name: str) -> dict:
    base = ROOT / "outputs" / "iclr27_phase6c" / "eval" / name
    out = {"name": name}
    sp = base / "strict" / "summary.json"
    if sp.exists():
        d = json.loads(sp.read_text())
        out["strict"] = {k: d["strict"].get(k) for k in STRICT_KEYS}
        out["legacy_first"] = {k: d["legacy_first_frame"].get(k) for k in LEGACY_KEYS}
        out["n_rows"] = d.get("n_rows")
        out["n_aligned_tracks"] = d.get("n_aligned_tracks")
    pp = base / "physical.json"
    if pp.exists():
        out["physical"] = json.loads(pp.read_text())
    cp = base / "calibration.json"
    if cp.exists():
        c = json.loads(cp.read_text())
        out["tau"] = c.get("tau")
        out["proxy_novel_acc"] = c.get("proxy_novel_acc")
    rp = base / "replay.log"
    if rp.exists():
        out["replay_log"] = rp.read_text().strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/iclr27_phase6c/eval/comparison.json")
    args = ap.parse_args()
    runs = [load_run(n) for n in args.names]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs, indent=2, default=float))
    print(f"wrote {out} ({len(runs)} runs)")
    for r in runs:
        s = r.get("strict", {})
        lf = r.get("legacy_first", {})
        print(f"{r['name']}: strict_known={s.get('known_occurrence_acc'):.3f} "
              f"birth={s.get('first_novel_birth_acc'):.3f} "
              f"reuse={s.get('novel_reuse_acc'):.3f} "
              f"cross={s.get('cross_physical_reuse_acc'):.3f} "
              f"nmi={s.get('novel_nmi'):.3f} ari={s.get('novel_ari'):.3f} "
              f"| legacy_known={lf.get('overall_known_acc'):.3f} "
              f"route={lf.get('route_aware_novel_acc'):.3f}")


if __name__ == "__main__":
    main()
