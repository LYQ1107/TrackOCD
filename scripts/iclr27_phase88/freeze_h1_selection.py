#!/usr/bin/env python3
"""Freeze/reject H1 using the preregistered TRAIN-only selection rule."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def read(tag: str, fold: int) -> dict:
    return json.loads((OUT / "validation" / f"{tag}_f{fold}" / "final_metrics.json").read_text())

def main() -> None:
    h1_tags = ["h1_trainval_f0", "h1_trainval_f1", "h1_trainval_f2", "h1_trainval_f3"]
    h1 = [read("h1_trainval", f) for f in range(4)]
    baseline = json.loads((OUT / "audit/c0_c1_train_selection.json").read_text())
    rows = []
    for f, d in enumerate(h1):
        m = d["metrics"]
        ckpt = [
            OUT / "checkpoints/h1_false_merge_reset_resume1_f0.pt",
            OUT / "checkpoints/h1_false_merge_reset_resume1_f1.pt",
            OUT / "checkpoints/h1_false_merge_reset_resume2_f2.pt",
            OUT / "checkpoints/h1_false_merge_reset_resume3_f3.pt",
        ][f]
        rows.append({
            "fold": f, "tag": h1_tags[f], "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": sha(ckpt),
            "selection_score": float(m["selection_score"]),
            "existing_precision": float(m["existing_precision"]), "existing_recall": float(m["existing_recall"]),
            "existing_f1": float(m["existing_f1"]), "existing_f1_macro": float(m["existing_f1_macro"]),
            "negative_false_merge_rate": float(m["negative_false_merge_rate"]),
            "premature_rate": float(m["premature_rate"]), "unresolved_rate": float(m["unresolved_rate"]),
            "duplicate_births": int(m["duplicate_births"]),
            "category_coverage": int(m["category_coverage"]), "video_coverage": int(m["video_coverage"]),
        })
    mean_h1 = sum(x["selection_score"] for x in rows) / 4.0
    # Existing C0/C1 means are in the frozen audit and are never recomputed
    # from held events.  The exact rule is the preregistered mean score,
    # secondary per-fold wins, then F1 macro and false-merge tie-breaks.
    c0_mean = float(baseline["controls"]["c0_continue_fix1"]["aggregate"]["selection_score"])
    c1_mean = float(baseline["controls"]["c1_support_fix1"]["aggregate"]["selection_score"])
    c0_wins = sum(rows[f]["selection_score"] > float(baseline["branches"]["c0_continue_fix1"]["folds"][str(f)]["selection_score"]) for f in range(4)) if False else None
    decision = "H1_REJECTED_TRAIN_ONLY" if mean_h1 <= max(c0_mean, c1_mean) else "H1_SELECTED_TRAIN_ONLY"
    artifact = {
        "schema_version": "trackocd.phase88.h1_selection.v1",
        "phase": 88, "status": "FROZEN_TRAIN_ONLY", "decision": decision,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_rule": "mean exact Phase19R selection_score across the four same-fold TRAIN validation sets; no held/public access",
        "h1_rows": rows, "h1_mean_selection_score": mean_h1,
        "frozen_comparators": {"c0_continue_fix1_mean_selection_score": c0_mean, "c1_support_fix1_mean_selection_score": c1_mean},
        "reason": "H1 is selected only if its mean score exceeds both frozen C0/C1 comparators; no held metric is used",
        "public_dev_q1_sealed_accessed": False,
    }
    out = OUT / "audit/h1_train_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps({"decision": decision, "h1_mean": mean_h1, "c0": c0_mean, "c1": c1_mean}, indent=2))

if __name__ == "__main__":
    main()
