#!/usr/bin/env python3
"""Select C0_CONTINUE or C1_SUPPORT using only the four TRAIN validations."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_mode(mode: str) -> list[dict]:
    rows = []
    for fold in range(4):
        tag = f"{mode}_f{fold}"
        train = OUT / "metrics" / f"{tag}.json"
        val = OUT / "validation" / f"{tag}_val" / "final_metrics.json"
        done = OUT / "completion" / f"{tag}.done"
        val_done = OUT / "validation" / f"{tag}_val" / "final.done"
        if not all(x.exists() for x in (train, val, done, val_done)):
            raise RuntimeError(f"incomplete control artifacts for {tag}")
        tr = json.loads(train.read_text())
        va = json.loads(val.read_text())
        if va.get("public_dev_q1_sealed_accessed") or va.get("future_rows_or_tracks") or va.get("ids_or_text_as_model_input"):
            raise RuntimeError(f"contract violation in {tag}")
        m = va["metrics"]
        rows.append({
            "fold": fold, "tag": tag,
            "train_metrics": str(train.resolve()), "train_metrics_sha256": sha(train),
            "validation_metrics": str(val.resolve()), "validation_metrics_sha256": sha(val),
            "checkpoint": tr["checkpoint"], "checkpoint_sha256": tr["checkpoint_sha256"],
            "updates": tr["updates"], "start_step": tr["start_step"], "support_mode": tr["support_mode"],
            "selection_score": m["selection_score"], "commit_ct": m["commit_ct"],
            "existing_precision": m["existing_precision"], "existing_recall": m["existing_recall"],
            "existing_f1": m["existing_f1"], "existing_f1_macro": m["existing_f1_macro"],
            "negative_false_merge_rate": m["negative_false_merge_rate"],
            "premature_rate": m["premature_rate"], "unresolved_rate": m["unresolved_rate"],
            "duplicate_births": m["duplicate_births"], "category_coverage": m["category_coverage"],
            "video_coverage": m["video_coverage"], "known_micro": m["known_micro"], "known_macro": m["known_macro"],
            "validation_events": va["events"],
        })
    return rows


def main() -> None:
    c0 = read_mode("c0_continue_fix1")
    c1 = read_mode("c1_support_fix1")
    per_fold_wins = sum(float(c0[i]["selection_score"]) > float(c1[i]["selection_score"]) for i in range(4))
    per_fold_ties = sum(float(c0[i]["selection_score"]) == float(c1[i]["selection_score"]) for i in range(4))
    def agg(rows: list[dict]) -> dict:
        keys = ["selection_score", "existing_precision", "existing_recall", "existing_f1", "existing_f1_macro", "negative_false_merge_rate", "premature_rate", "unresolved_rate", "known_micro", "known_macro"]
        return {k: sum(float(x[k]) for x in rows) / 4.0 for k in keys}
    a0, a1 = agg(c0), agg(c1)
    selected = "c0_continue_fix1" if a0["selection_score"] >= a1["selection_score"] else "c1_support_fix1"
    selected_rows = c0 if selected == "c0_continue_fix1" else c1
    payload = {
        "schema_version": "trackocd.phase88.c0_c1_train_selection.v1",
        "phase": 88, "status": "FROZEN_TRAIN_ONLY",
        "selection_rule": {
            "primary": "mean exact Phase19R selection_score across four same-fold TRAIN_disjoint validation results",
            "secondary": "number of folds with strictly higher selection_score",
            "tie_break": "mean existing_f1_macro, then lower mean negative_false_merge_rate",
            "held_or_public_metrics_used": False,
        },
        "controls": {
            "c0_continue_fix1": {"folds": c0, "aggregate": a0},
            "c1_support_fix1": {"folds": c1, "aggregate": a1},
        },
        "per_fold_c0_wins": per_fold_wins,
        "per_fold_ties": per_fold_ties,
        "selected_branch": selected,
        "selected_fold_checkpoints": selected_rows,
        "event_tag": "fix2", "additional_updates": 10000, "base_step": 20000,
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out = OUT / "audit/c0_c1_train_selection.json"
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
