"""Phase 6B Audit 3 — CSV -> strict evaluator wiring (deterministic).

Two synthetic streams on the frozen Q1 dev rows (no model involved):
  - correct: every aligned row gets KNOWN(c) with c = GT category;
  - wrong:   every aligned row gets KNOWN(c') with c' != GT category.
The strict evaluator must report known_occurrence_acc == 1.0 and 0.0
respectively. This proves the CSV `sem_action`/`sem_sid` -> evaluator
expected-category chain is correctly wired.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import (
    group_tracks,
    load_gt_tracks_dev,
    load_proposals,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
EVAL = ROOT / "src/iclr27_phase5a/evaluation/strict_causal_eval.py"


def make_csv(src_csv: Path, out_csv: Path, mode: str):
    rows = load_proposals(src_csv)
    stream, labels_all = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels_all[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(group_tracks(rows), gb)
    for r in rows:
        key = (int(r["video_id"]), int(r["track_id"]))
        sid = mapping.get(key)
        if sid is None:
            continue
        lab = labels[sid]
        if lab["protocol_role"] not in ("supported_known", "zero_shot_known"):
            continue
        c = int(lab["ground_truth_category_id"])
        if mode == "correct":
            r["sem_action"] = "known"
            r["sem_sid"] = str(c)
        else:
            wrong = (c % 1203) + 1
            if wrong == c:
                wrong = (c % 1203) + 2
            r["sem_action"] = "known"
            r["sem_sid"] = str(wrong)
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_eval(csv_path: Path, feats_path: Path, out_dir: Path):
    cmd = [
        PY, str(EVAL),
        "--proposals", str(csv_path),
        "--feats", str(feats_path),
        "--proto-dir", "outputs/iclr27_phase5a/pilot/episodes",
        "--embed", "h", "--mode", "jointcsv", "--filter", "aligned",
        "--device", "cuda:0", "--out", str(out_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True,
                   capture_output=True, text=True)
    return json.loads((out_dir / "summary.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-csv",
                    default=str(ROOT / "outputs/iclr27_phase6a/q1/main_final/proposals_dev.csv"))
    ap.add_argument("--feats",
                    default=str(ROOT / "outputs/iclr27_phase6a/q1/main_final/feats.npz"))
    ap.add_argument("--out", default=str(ROOT / "outputs/iclr27_phase6b/audit/evaluator_wiring"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.src_csv)
    feats = Path(args.feats)
    correct_csv = out / "synthetic_correct.csv"
    wrong_csv = out / "synthetic_wrong.csv"
    shutil.copy(feats, out / "feats.npz")
    make_csv(src, correct_csv, "correct")
    make_csv(src, wrong_csv, "wrong")

    r_correct = run_eval(correct_csv, out / "feats.npz", out / "correct")
    r_wrong = run_eval(wrong_csv, out / "feats.npz", out / "wrong")
    acc_correct = r_correct["strict"]["known_occurrence_acc"]
    acc_wrong = r_wrong["strict"]["known_occurrence_acc"]
    result = {
        "known_occurrence_acc_correct": acc_correct,
        "known_occurrence_acc_wrong": acc_wrong,
        "wiring_ok": abs(acc_correct - 1.0) < 1e-9 and abs(acc_wrong) < 1e-9,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["wiring_ok"]:
        raise SystemExit("EVALUATOR_WIRING_FAILED")
    print("EVALUATOR_WIRING_OK")


if __name__ == "__main__":
    main()
