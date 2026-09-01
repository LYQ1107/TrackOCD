"""Apply the preregistered public safety gate after public truth scoring."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


OUT = Path("outputs/iclr27_phase19r")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=OUT / "metrics/public_gate.json"); a = p.parse_args()
    public = json.loads((OUT / "metrics/public_after_freeze.json").read_text())
    known = json.loads((OUT / "metrics/known_after_freeze.json").read_text())
    known_rows = known.get("candidates", {})
    candidates: dict[str, Any] = {}
    for name, payload in public.get("candidates", {}).items():
        m = payload.get("metrics", {}); km = known_rows.get(name, {})
        checks = {"commit_ct": int(m.get("commit_ct", {}).get("correct", 0)) >= 15,
                  "category_coverage": int(m.get("category_coverage", 0)) >= 5,
                  "video_coverage": int(m.get("video_coverage", 0)) >= 8,
                  "existing_precision": float(m.get("existing_precision", 0.0)) >= .70,
                  "negative_false_merge": float(m.get("negative_false_merge_rate", 1.0)) <= .15,
                  "known_micro": float(km.get("micro_accuracy", 0.0)) >= .206,
                  "known_macro": float(km.get("category_macro_accuracy", 0.0)) >= .139}
        candidates[name] = {"checks": checks, "gate_pass": bool(all(checks.values())),
                            "public_metrics": m, "known_metrics": {k: km.get(k) for k in ("micro_accuracy", "category_macro_accuracy", "rows", "tracks", "categories")}}
    result = {"protocol": "trackocd_iclr27_phase19r_public_safety_gate", "truth_join_after_freeze": True,
              "gate_thresholds": {"commit_ct": "15/41", "category_coverage": 5, "video_coverage": 8,
                                   "existing_precision": .70, "negative_false_merge": .15,
                                   "known_micro": .206, "known_macro": .139},
              "candidates": candidates,
              "primary_gate_pass": bool(candidates.get("main", {}).get("gate_pass", False)),
              "devplus_q1_open": bool(candidates.get("main", {}).get("gate_pass", False))}
    a.parent.mkdir(parents=True, exist_ok=True); tmp = a.with_name(a.name + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, a)
    print(json.dumps({k: v["gate_pass"] for k, v in candidates.items()}, sort_keys=True))


if __name__ == "__main__": main()
