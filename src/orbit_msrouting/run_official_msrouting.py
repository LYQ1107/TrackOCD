"""Run a frozen ORBIT-MSRouting candidate on official Pure Full seed1027.

The candidate config (checkpoint, gate mode, state features, thresholds)
must already exist in outputs/orbit_msrouting/frozen_candidates/.  This
script never tunes anything from official output; it only executes the
frozen config and records the result.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit_msrouting.evaluate_msrouting import (
    bucket_rows,
    evaluate_official_msrouting,
    load_msrouting_checkpoint,
    result_row,
    write_csv,
)


def memory_stats(logs):
    by_vid = defaultdict(list)
    for l in logs:
        if l["predicted_action"] in ("EXISTING_NOVEL", "NEW_NOVEL"):
            by_vid[l["predicted_virtual_novel_id"]].append(l)
    hubs = sum(1 for ls in by_vid.values()
               if len({l["class"] for l in ls}) >= 2)
    known_origin = sum(1 for ls in by_vid.values()
                       if ls[0]["role"] == "known")
    return {"hub_prototype_count": hubs,
            "known_origin_prototype_count": known_origin,
            "predicted_prototype_count": len(by_vid)}


def load_frozen(candidate):
    p = ROOT / "outputs/orbit_msrouting/frozen_candidates" / \
        f"candidate_{candidate.lower()}.json"
    return json.loads(p.read_text())


def csv_text(rows):
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=["A", "B"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--diagnostic_name", default=None,
                    help="optional override row name (diagnostic runs only)")
    args = ap.parse_args()
    cfg = load_frozen(args.candidate)
    model, ck = load_msrouting_checkpoint(
        str(ROOT / cfg["checkpoint"]), args.device)
    logs, gt = evaluate_official_msrouting(
        model, ck, args.device,
        gate_thr=cfg.get("gate_threshold", 0.5),
        compat_thr=cfg.get("compat_threshold", 0.45),
        compat_margin=cfg.get("compat_margin", 0.05))
    name = args.diagnostic_name or f"ORBIT_MSRouting_{cfg['candidate']}"
    row = result_row(logs, gt, name)
    row.update(memory_stats(logs))
    row.update({"gate_mode": cfg.get("gate_mode"),
                "state_feats": ",".join(cfg.get("state_feats", [])),
                "gate_threshold": cfg.get("gate_threshold"),
                "compat_threshold": cfg.get("compat_threshold"),
                "compat_margin": cfg.get("compat_margin"),
                "checkpoint_sha256": cfg.get("checkpoint_sha256"),
                "frozen_at": cfg.get("frozen_at")})
    print(json.dumps(row, indent=1), flush=True)
    out_dir = ROOT / "outputs/orbit_msrouting/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"{name}_seed1027.csv", [row])
    write_csv(out_dir / f"{name}_seed1027_buckets.csv",
              bucket_rows(logs, gt, name))
    write_csv(out_dir / f"{name}_seed1027_trajectory.csv", logs)
    comp_path = out_dir / "official_comparison.csv"
    rows = []
    if comp_path.exists():
        rows = list(csv.DictReader(comp_path.open()))
    rows = [r for r in rows if r.get("candidate") != name]
    rows.append(row)
    comp_path.write_text(csv_text(rows))
    print("saved", out_dir)


if __name__ == "__main__":
    main()
