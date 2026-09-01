"""Verify the Phase18 B1 exact interface without mutating historical files."""
from __future__ import annotations

import json
from pathlib import Path

from src.iclr27_phase18.evaluation.baseline_b1 import B1Policy, event_metrics, load_data, simulate_event, known_metrics


def main() -> None:
    data = load_data(); policies = {f["fold"]: B1Policy(data, f) for f in data["folds"]["folds"]}
    records = [simulate_event(policies[e["fold"]], e) for e in data["positives"] + data["negatives"]]
    m = event_metrics(records); known = [known_metrics(policies[f]) for f in sorted(policies)]
    historical = json.loads(Path("outputs/iclr27_phase18/eval/b1_prereg_baseline.json").read_text())
    expected = historical["metrics"]
    checks = {
        "commit_ct": (m["commit_ct"]["correct"], expected["commit_ct"]["correct"]),
        "commit_eligible": (m["commit_ct"]["eligible"], expected["commit_ct"]["eligible"]),
        "existing_precision": (m["existing_precision"], expected["existing_precision"]),
        "negative_false_merge_rate": (m["negative_false_merge_rate"], expected["negative_false_merge_rate"]),
        "known_micro_mean": (sum(x["micro_accuracy"] for x in known) / len(known), historical["known"]["micro_accuracy_mean"]),
        "known_macro_mean": (sum(x["category_macro_accuracy"] for x in known) / len(known), historical["known"]["category_macro_accuracy_mean"]),
    }
    passed = all(a == b if isinstance(a, int) else abs(a - b) < 1e-12 for a, b in checks.values())
    result = {"protocol": "trackocd_iclr27_phase19r_phase18_b1_exact_interface_parity", "passed": passed, "checks": {k: {"replayed": a, "historical": b, "delta": (a - b) if not isinstance(a, int) else a - b} for k, (a, b) in checks.items()}, "historical_artifact": "outputs/iclr27_phase18/eval/b1_prereg_baseline.json", "historical_files_unchanged": True}
    p = Path("outputs/iclr27_phase19r/metrics/b1_parity.json"); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, indent=2, sort_keys=True))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
