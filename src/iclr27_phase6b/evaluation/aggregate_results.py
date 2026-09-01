"""Aggregate all Phase 6B evaluation artifacts for one model name into a
compact JSON used by the final report builder."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/iclr27_phase6b")

    result = {"name": args.name}
    stats_path = root / "q1" / args.name / "joint_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        actions = Counter(r["action"] for r in stats)
        decision3 = [r["decision_logits3"] for r in stats
                     if r.get("decision_logits3")]
        result["semantic_stream"] = {
            "n_rows": len(stats),
            "actions": dict(actions),
            "n_known": actions.get("known", 0),
            "n_new": actions.get("new", 0),
            "n_existing": actions.get("existing", 0),
            "n_novel_slots": max((r["sid"] + 1 for r in stats
                                  if r["action"] in ("new", "existing")),
                                 default=0),
            "mean_new_logit": float(sum(d[2] for d in decision3) /
                                    len(decision3)) if decision3 else None,
            "mean_existing_logit": float(sum(d[1] for d in decision3) /
                                         len(decision3)) if decision3 else None,
            "mean_known_logit": float(sum(d[0] for d in decision3) /
                                      len(decision3)) if decision3 else None,
        }
    strict = root / "strict_eval" / f"{args.name}_dsct" / "summary.json"
    if strict.exists():
        result["strict"] = json.loads(strict.read_text()).get("strict", {})
    legacy = root / "strict_eval" / f"{args.name}_dsct" / "summary.json"
    if legacy.exists():
        s = json.loads(legacy.read_text())
        result["legacy_first_frame"] = s.get("legacy_first_frame", {})
    phys = root / "physical_eval" / f"{args.name}.json"
    if phys.exists():
        result["physical"] = json.loads(phys.read_text())
    obj = root / "strict_eval" / f"{args.name}_objectness_audit.json"
    if obj.exists():
        result["objectness"] = json.loads(obj.read_text())
    contract = root / "strict_eval" / f"{args.name}_causal_contract.json"
    if contract.exists():
        result["contract"] = json.loads(contract.read_text())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
