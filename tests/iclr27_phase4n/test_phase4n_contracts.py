"""Phase 4N contract tests (real risks only)."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_corrected_heldout_gt_has_real_categories():
    d = json.loads((ROOT / "outputs" / "iclr27_phase4n" / "audit" /
                    "validation_heldout_tao_corrected.json").read_text())
    cats = Counter(a["category_id"] for a in d["annotations"])
    assert len(cats) >= 20
    known = set(json.loads((ROOT / "data" / "trackocd_v1" / "pure" /
                            "splits" /
                            "supported_known_ids.json").read_text()))
    known_anns = sum(v for k, v in cats.items() if k in known)
    assert known_anns > 0
    assert len(d["images"]) == 887


def test_heldout_gt_not_used_for_tuning():
    """No phase4n method config may read held-out GT."""
    for p in (ROOT / "src" / "iclr27_phase4n").rglob("*.py"):
        if p.name in ("eval_heldout_corrected.py",
                      "build_detection_population.py",
                      "correct_heldout_gt.py",
                      "write_results_docs.py"):
            continue
        text = p.read_text()
        assert "validation_heldout_tao_corrected" not in text


def test_detection_population_present():
    for name in ("detection_population_dev.csv",
                 "detection_population_heldout_corrected.csv"):
        p = ROOT / "outputs" / "iclr27_phase4n" / "audit" / name
        assert p.exists() and p.stat().st_size > 100000
    rows = list(csv.DictReader(open(
        ROOT / "outputs" / "iclr27_phase4n" / "audit" /
        "detection_population_dev.csv")))
    roles = Counter(r["gt_role"] for r in rows)
    assert roles.get("known", 0) > 0
    assert roles.get("novel", 0) > 0
    assert roles.get("fp", 0) > roles.get("known", 0)


def test_audit_csvs_present():
    names = ["detector_score_distributions.csv",
             "detector_threshold_curve.csv", "persistent_fp_features.csv",
             "validity_predictability.csv", "gate_scores_dev.csv",
             "gate_scores_heldout.csv", "gate_shift_by_age.csv",
             "gate_shift_by_video.csv", "detector_gate_interaction.csv",
             "gate_shift_summary.csv"]
    for n in names:
        assert (ROOT / "outputs" / "iclr27_phase4n" / "audit" /
                n).exists()


def test_github_commits_recorded():
    rows = list(csv.DictReader(open(
        ROOT / "outputs" / "iclr27_phase4n" / "open_source" /
        "repository_inventory.csv")))
    assert len(rows) >= 6
    for r in rows:
        if r["commit"]:
            assert len(r["commit"]) == 40
        assert r["license"]
