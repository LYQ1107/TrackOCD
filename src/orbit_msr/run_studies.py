"""Long-stream proxy evaluation for ORBIT-MSR mechanism studies."""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.orbit_msr.evaluate import load_msr_model, evaluate_long_stream

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "orbit_msr" / "meta_dev"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    configs = [
        ("FC", "runs/orbit_fc/fc_F1/model.pth", 0.5, 0.5),
        ("FC", "runs/orbit_fc/fc_F1/model.pth", 0.5, 0.45),
        ("KG1", "runs/orbit_msr/msr_kg1/model.pth", 0.5, 0.5),
        ("KG1", "runs/orbit_msr/msr_kg1/model.pth", 0.5, 0.45),
        ("KG2", "runs/orbit_msr/msr_kg2/model.pth", 0.5, 0.5),
        ("KG2", "runs/orbit_msr/msr_kg2/model.pth", 0.5, 0.45),
        ("NR1", "runs/orbit_msr/msr_nr1/model.pth", 0.5, 0.5),
        ("NR1", "runs/orbit_msr/msr_nr1/model.pth", 0.5, 0.45),
        ("NR2", "runs/orbit_msr/msr_nr2/model.pth", 0.5, 0.5),
        ("NR2", "runs/orbit_msr/msr_nr2/model.pth", 0.5, 0.45),
        ("T2", "runs/orbit_msr/msr_t2/model.pth", 0.5, 0.5),
        ("T2", "runs/orbit_msr/msr_t2/model.pth", 0.5, 0.45),
    ]
    all_rows = []
    for name, path, gt, rt in configs:
        model, ck = load_msr_model(ROOT / path)
        rows_out, _ = evaluate_long_stream(model, ck, "cuda", gate_thr=gt,
                                           reuse_thr=rt)
        for r in rows_out:
            r["ckpt"] = name
            r["gate_thr"] = gt
            r["reuse_thr"] = rt
            all_rows.append(r)
            if r["scope"] == "overall":
                print({k: r[k] for k in ["ckpt", "gate_thr", "reuse_thr",
                                         "known_acc", "rn_acc", "cond_novel_acc",
                                         "routing_recall", "ari", "count_error",
                                         "known_to_novel", "novel_to_known",
                                         "repeated_false_birth", "wrong_existing",
                                         "first_merge"]}, flush=True)
    keys = list(all_rows[0].keys())
    with open(OUT / "long_stream_all_configs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    def extract(csv_path, names):
        rows = [r for r in all_rows if r["ckpt"] in names and r["scope"] == "overall"]
        out = []
        for r in rows:
            out.append({k: r[k] for k in
                        ["ckpt", "gate_thr", "reuse_thr", "all_acc", "known_acc",
                         "rn_acc", "cond_novel_acc", "routing_recall", "nmi", "ari",
                         "count_error", "known_to_novel", "novel_to_known",
                         "repeated_false_birth", "wrong_existing", "first_merge"]})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)

    extract(OUT / "known_gate_comparison.csv", ["FC", "KG1", "KG2"])
    extract(OUT / "novel_reuse_comparison.csv", ["FC", "NR1", "NR2"])
    extract(OUT / "calibration_comparison.csv", ["FC", "KG1", "T2"])


if __name__ == "__main__":
    main()
