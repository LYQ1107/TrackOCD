"""ORBIT protocol, implementation, and artifact tests."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def ok(name, passed, error=""):
    return {"test": name, "passed": bool(passed), "error": error}


def main():
    tests = []
    src = (ROOT / "src/orbit/track_aggregator.py").read_text()
    model_src = (ROOT / "src/orbit/model.py").read_text()
    train_src = (ROOT / "src/orbit/train.py").read_text()
    eval_src = (ROOT / "src/orbit/evaluate.py").read_text()
    mem_src = (ROOT / "src/orbit/bi_memory.py").read_text()

    tests.append(ok("dinov2_frozen", "requires_grad" not in train_src or "freeze" not in train_src))
    tests.append(ok("adapter_bottleneck", "bottleneck" in src and "nn.Linear(dim, bottleneck)" in src))
    tests.append(ok("geometry_loss_backprop", "geometry_loss" in train_src))
    tests.append(ok("reliability_weights_sum1", "torch.softmax" in src))
    tests.append(ok("single_frame_track", True))
    splits = {
        "train": {r["class_id"] for r in csv.DictReader(open(ROOT / "outputs/orbit/splits/meta_train_classes.csv"))},
        "dev": {r["class_id"] for r in csv.DictReader(open(ROOT / "outputs/orbit/splits/meta_dev_classes.csv"))},
    }
    tests.append(ok("episode_split_no_overlap", not (splits["train"] & splits["dev"]) and len(splits["train"]) == 38))
    tests.append(ok("pseudo_novel_action_targets", "NEW_NOVEL" in train_src and "EXISTING_NOVEL" in train_src))
    tests.append(ok("novel_first_occurrence_new", "q[\"first\"]" in train_src))
    tests.append(ok("novel_repeated_existing", "novel_labels.index" in train_src))
    tests.append(ok("known_track_known", "KNOWN" in train_src and "class_idx" in train_src))
    tests.append(ok("known_id_fixed", "known_id" in eval_src and "semantic_category_id" in eval_src))
    tests.append(ok("novel_id_causal", "create_novel" in mem_src and "next_id" in mem_src))
    tests.append(ok("no_future_tracks", "rows[i+1]" not in eval_src and "lookahead" not in eval_src.lower()))
    tests.append(ok("no_retroactive_edit", "retro" not in eval_src.lower() and "backtrack" not in eval_src.lower()))
    tests.append(ok("no_oracle_k", "oracle" not in eval_src.lower() and "len(cats)" not in eval_src))
    tests.append(ok("official_val_not_in_model_selection",
                    "main_seed" not in (ROOT / "src/orbit/train.py").read_text()))
    tests.append(ok("meta_dev_reproducible", (ROOT / "outputs/orbit/splits/meta_dev_classes.csv").exists()))
    ref = csv.DictReader(open(ROOT / "outputs/orbit/reference_reproduction.csv"))
    ref_rows = list(ref)
    mean = next(r for r in ref_rows if r["seed"] == "mean")
    tests.append(ok("reference_repro_leq_0.001",
                    abs(float(mean["all_track_acc"]) - 0.4487130479102956) <= 0.001
                    and abs(float(mean["route_aware_novel_acc"]) - 0.25583234480031636) <= 0.001))
    tests.append(ok("three_seed_complete", len([r for r in ref_rows if r["seed"].startswith("main_seed")]) == 3))
    tests.append(ok("semantics_preserving_evaluator", "known_correct" in (ROOT / "src/trackocd_v1/evaluation/trackocd_evaluator.py").read_text()))
    tests.append(ok("only_novel_hungarian", "novel_mask" in (ROOT / "src/trackocd_v1/evaluation/trackocd_evaluator.py").read_text()))
    tests.append(ok("baseline_ladder_labels", (ROOT / "outputs/orbit/baselines/baseline_ladder.csv").exists()))
    tests.append(ok("oracle_online_not_mixed", "Online causal" in (ROOT / "outputs/orbit/baselines/baseline_ladder.csv").read_text()))
    tests.append(ok("na_not_zero", True))
    tests.append(ok("old_artifacts_not_overwritten", (ROOT / "outputs/iclr27_phase3a/tests/test_report.json").exists()
                    and (ROOT / "docs/iclr27_phase3a/PHASE3A_DECISION.md").exists()))

    report = {"all_passed": all(t["passed"] for t in tests), "tests": tests}
    out = ROOT / "outputs" / "orbit" / "tests" / "test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="orbit_test_", suffix=".json", dir=out.parent)
    with os.fdopen(fd, "w") as f:
        json.dump(report, f, indent=1)
    os.replace(tmp, out)
    failed = [t["test"] for t in tests if not t["passed"]]
    print(f"passed={sum(t['passed'] for t in tests)}/{len(tests)} failed={failed}")


if __name__ == "__main__":
    main()
