"""Check that frozen public prediction JSON contains no evaluator truth fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN = {"category_gt_denominator_only", "target_category_gt_denominator_only", "distractor_category_gt_denominator_only", "kind", "oracle_birth_category", "gt_category_evaluator_only", "correct"}


def walk(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield path + "." + k, k
            yield from walk(v, path + "." + k)
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from walk(v, f"{path}[{i}]")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, default=Path("outputs/iclr27_phase19r/audit/frozen_leakage.json")); a = p.parse_args()
    marker = Path("outputs/iclr27_phase19r/completion/public_predictions.frozen"); paths = sorted(Path("outputs/iclr27_phase19r/public_predictions").glob("*_raw.json"))
    bad = []
    for path in paths:
        obj = json.loads(path.read_text())
        bad.extend({"file": str(path), "path": loc, "field": key} for loc, key in walk(obj) if key in FORBIDDEN)
    result = {"protocol": "trackocd_iclr27_phase19r_frozen_prediction_leakage_check", "marker_present": marker.exists(), "files": [str(p) for p in paths], "forbidden_fields": sorted(FORBIDDEN), "violations": bad, "passed": bool(marker.exists() and paths and not bad)}
    a.parent.mkdir(parents=True, exist_ok=True); a.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"passed": result["passed"], "files": len(paths), "violations": len(bad)}, sort_keys=True))


if __name__ == "__main__": main()
