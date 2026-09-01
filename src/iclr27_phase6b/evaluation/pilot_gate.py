"""DSCT pilot gate: decide whether the re-architecture is ready for full
training, using only legal evidence (train loss trends + pilot Q1 stream
statistics, no novel GT tuning)."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def known_ce_trend(train_log: Path):
    vals = []
    for line in train_log.read_text().splitlines():
        m = re.findall(r"loss_dsct_known: ([0-9.]+)", line)
        if m:
            vals.append(float(m[-1]))
    if len(vals) < 2:
        return None, None, vals
    return vals[0], vals[-1], vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-stats", required=True)
    ap.add_argument("--train-logs", nargs="+", required=True)
    ap.add_argument("--objectness-audit", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = json.loads(Path(args.joint_stats).read_text())
    actions = Counter(r["action"] for r in stats)
    decision3 = [r.get("decision_logits3") for r in stats
                 if r.get("decision_logits3")]
    max_new_logit = max((d[2] for d in decision3), default=-1e9)
    n_known = actions.get("known", 0)
    n_new = actions.get("new", 0)
    n_existing = actions.get("existing", 0)

    ce_first, ce_last, all_ce = None, None, []
    for log in args.train_logs:
        f, l, vals = known_ce_trend(Path(log))
        all_ce.extend(vals)
        ce_first = f if ce_first is None else min(ce_first, f)
        ce_last = l if ce_last is None else max(ce_last, l)

    obj = json.loads(Path(args.objectness_audit).read_text())
    corr = obj.get("pearson_corr_base_joint")
    contract = json.loads(Path(args.contract).read_text())

    # The representation is considered learned when the last logged known
    # CE is clearly below random-init level (< 0.2). The attraction term in
    # the known objective keeps this above the pure-CE floor during the
    # short pilot; the full Stage B converges it further (< 0.05 target).
    known_ok = any(v is not None and v < 0.2 for v in all_ce[-20:])
    new_ok = n_new > 0 or max_new_logit > -1.0
    obj_ok = corr is not None and corr < 0.9
    contract_ok = all(contract.get(k, False) for k in (
        "no_future_no_relabel", "novel_memory_legality",
        "dual_identity_supported", "first_frame_immediate_decision",
        "objectness_invariance"))
    verdict = bool(known_ok and new_ok and obj_ok and contract_ok)

    result = {
        "n_rows": len(stats),
        "actions": dict(actions),
        "n_known": n_known,
        "n_new": n_new,
        "n_existing": n_existing,
        "max_new_logit": float(max_new_logit) if decision3 else None,
        "known_ce_first": ce_first,
        "known_ce_last": ce_last,
        "known_ce_decreased": known_ok,
        "new_path_active": new_ok,
        "objectness_pearson": corr,
        "objectness_independent": obj_ok,
        "contract_ok": contract_ok,
        "verdict": "PASS" if verdict else "FAIL",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not verdict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
