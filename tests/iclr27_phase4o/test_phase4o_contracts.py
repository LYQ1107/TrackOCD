"""Phase 4O contract tests (real risks only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_corrected_gt_used_everywhere():
    """Phase 4O detector evaluation must use the corrected 28-cat GT."""
    d = json.loads((ROOT / "outputs" / "iclr27_phase4n" / "audit" /
                    "validation_heldout_tao_corrected.json").read_text())
    cats = {a["category_id"] for a in d["annotations"]}
    assert len(cats) >= 20
    # old collapsed GT must not appear in phase4o scripts
    for p in (ROOT / "src" / "iclr27_phase4o").rglob("*.py"):
        assert "val_split/all.json" not in p.read_text()


def test_detector_only_proposals_valid():
    for name in ("proposals_yoloe_dev.csv", "proposals_wedetect_dev.csv"):
        p = ROOT / "outputs" / "iclr27_phase4o" / "detector_only" / name
        rows = list(csv.DictReader(open(p)))
        assert len(rows) > 1000
        scores = [float(r["score"]) for r in rows[:500]]
        assert all(0.0 <= s <= 1.0 for s in scores)
        for r in rows[:10]:
            bb = json.loads(r["bbox_xyxy"])
            assert len(bb) == 4
            assert bb[2] > bb[0] and bb[3] > bb[1]


def test_pareto_csvs_present():
    for n in ("summary.csv", "novel_recall_fp_curve.csv",
              "fixed_fp_comparison.csv", "fixed_novel_recall.csv",
              "proposal_budget.csv"):
        assert (ROOT / "outputs" / "iclr27_phase4o" / "detector_only" /
                n).exists()


def test_no_trackocd_reentry_without_pass():
    """No T0-T3 runs may exist because no detector passed the gate."""
    decision = (ROOT / "docs" / "iclr27_phase4o" /
                "DETECTOR_SELECTION_DECISION.md").read_text()
    assert "NO_DETECTOR_FRONTEND_CLEAR_PROGRESS" in decision
    dev = ROOT / "outputs" / "iclr27_phase4o" / "dev"
    assert not (dev / "tracking_results.csv").exists()


def test_open_source_commits_and_licenses():
    rows = list(csv.DictReader(open(
        ROOT / "outputs" / "iclr27_phase4o" / "open_source" /
        "repository_inventory.csv")))
    assert len(rows) >= 4
    for r in rows:
        if r["commit"]:
            assert len(r["commit"]) == 40
        assert r["license"]


def test_old_outputs_preserved():
    p = ROOT / "outputs" / "iclr27_phase4n" / "audit" / \
        "validation_heldout_tao_corrected.json"
    assert p.exists()
    q = ROOT / "outputs" / "iclr27_phase4m" / "runs" / "dev" / \
        "trackeval" / "tracking_j1b.json"
    assert q.exists()
