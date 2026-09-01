"""Controlled routing cases: matched visual evidence, different memory state."""
from __future__ import annotations

import csv

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def load(stream):
    rows = list(csv.DictReader(open(
        f"{ROOT}/outputs/iclr27_phase4f/audit/memory_trajectory_m2_{stream}.csv")))
    return [r for r in rows if r["true_role"] == "novel"]


def bucket(m):
    m = int(m)
    if m < 33:
        return "0-32"
    if m < 129:
        return "33-128"
    if m < 257:
        return "129-256"
    return "257+"


def main():
    import pathlib
    out = pathlib.Path(f"{ROOT}/outputs/iclr27_phase4g/audit")
    out.mkdir(parents=True, exist_ok=True)
    rows = load("official")
    cases = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if abs(float(a["known_best_sim"]) - float(b["known_best_sim"])) > 0.02:
                continue
            if abs(float(a["known_margin"]) - float(b["known_margin"])) > 0.02:
                continue
            ba, bb = bucket(a["memory_size"]), bucket(b["memory_size"])
            if ba == bb:
                continue
            cases.append({
                "sample_a": a["sample_id"], "bucket_a": ba,
                "gate_a": round(float(a["gate_prob"]), 3),
                "action_a": a["predicted_action"],
                "sample_b": b["sample_id"], "bucket_b": bb,
                "gate_b": round(float(b["gate_prob"]), 3),
                "action_b": b["predicted_action"],
                "sim": round(float(a["known_best_sim"]), 3),
                "margin": round(float(a["known_margin"]), 3),
                "memory_a": a["memory_size"], "memory_b": b["memory_size"],
                "gate_delta": round(float(b["gate_prob"]) - float(a["gate_prob"]), 3),
            })
    cases.sort(key=lambda c: -abs(c["gate_delta"]))
    with open(f"{out}/controlled_routing_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        w.writeheader()
        w.writerows(cases[:500])
    print("matched pairs", len(cases))
    for c in cases[:8]:
        print(c)


if __name__ == "__main__":
    main()
