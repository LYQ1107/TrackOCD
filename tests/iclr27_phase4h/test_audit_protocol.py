"""Real-risk tests for the Phase 4H root-cause audit pipeline."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
AUDIT = ROOT / "outputs" / "iclr27_phase4h" / "audit"


def _rows_synthetic():
    rows = []
    for i in range(30):
        rows.append({
            "sample_id": f"s{i}",
            "role": "known" if i < 12 else "novel",
            "class": 100 + (i % 5),
            "first_occurrence": i % 5 == 0,
        })
    return rows


def test_permutation_is_a_reshuffle_only():
    from src.iclr27_phase4h.audit_permutations import (
        build_orders,
        novel_subsequence_permuted,
    )
    rows = _rows_synthetic()
    hardness = {c: c / 100.0 for c in range(100, 105)}
    orders = build_orders(rows, hardness, [1001, 2002])
    assert [m for m, _, _ in orders] == ["P0", "P1", "P2", "P1", "P2",
                                         "P3", "P4"]
    for mode, seed, order in orders:
        assert len(order) == len(rows)
        assert Counter(r["sample_id"] for r in order) == Counter(
            r["sample_id"] for r in rows)
        # same row objects: features/GT/class identity cannot change
        assert all(any(r is o for r in rows) for o in order)
    # P3 must be hardest-first, P4 easiest-first at the class-block level
    p3 = [o for m, _, o in orders if m == "P3"][0]
    p4 = [o for m, _, o in orders if m == "P4"][0]
    novel_p3 = [r["class"] for r in p3 if r["role"] == "novel"]
    novel_p4 = [r["class"] for r in p4 if r["role"] == "novel"]
    assert novel_p3[0] == max(hardness, key=hardness.get)
    assert novel_p4[0] == min(hardness, key=hardness.get)


def test_permutation_results_csv_complete():
    rows = list(csv.DictReader((AUDIT / "permutation_results.csv").open()))
    assert len(rows) == 13  # P0 + 5xP1 + 5xP2 + P3 + P4
    modes = Counter(r["mode"] for r in rows)
    assert modes["P0"] == 1 and modes["P1"] == 5 and modes["P2"] == 5
    assert modes["P3"] == 1 and modes["P4"] == 1
    for r in rows:
        for key in ("rn_acc", "ari", "novel_to_known", "final_memory_size"):
            assert key in r and r[key] != ""


def test_root_cause_probe_coverage():
    rows = list(csv.DictReader((AUDIT / "root_cause_probe.csv").open()))
    names = {r["model"] for r in rows}
    assert {"hardness_only", "hardness+position",
            "hardness+position+memory", "memory_only"} <= names
    by = {r["model"]: float(r["auc"]) for r in rows}
    assert by["hardness_only"] > by["memory_only"]
    assert by["hardness+position+memory"] >= by["hardness+position"] - 1e-9


def test_counterfactual_replay_keeps_query_fixed():
    rows = list(csv.DictReader(
        (AUDIT / "counterfactual_memory_replay.csv").open()))
    by = {}
    for r in rows:
        by.setdefault(r["sample_id"], []).append(r)
    assert len(by) == 843
    for q, rs in by.items():
        assert len(rs) == 4
        assert {r["snapshot"] for r in rs} == {"mem32", "mem128",
                                               "mem256", "mem400"}
        # frozen evidence and actual action must be snapshot-invariant
        assert len({r["best_known_sim"] for r in rs}) == 1
        assert len({r["known_margin"] for r in rs}) == 1
        assert len({r["actual_action"] for r in rs}) == 1


def test_repo_pins_are_real_and_licenses_recorded():
    inv = list(csv.DictReader(
        (ROOT / "outputs" / "iclr27_phase4h" / "open_source"
         / "repository_inventory.csv").open()))
    assert len(inv) == 6
    for row in inv:
        assert row["Commit"] and row["License"]
        repo = ROOT / "third_party" / "research_refs_phase4h" / {
            "W-DOE": "W-DOE",
            "HAAML": "HardnessAwareMargin",
            "MEPU-OWOD": "mepu-owod",
            "SphOR": "SphOR",
            "WSOE": "WSOE",
            "Adversarial-OE": "Adversarial-OE",
        }[row["Method"]]
        assert (repo / ".git").exists()
        head_sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        assert head_sha == row["Commit"]
    assert any(r["Year"] == "2026" for r in inv)
    assert sum(r["Year"] == "2025" for r in inv) >= 3


def test_input_manifest_hash_matches_checkpoint():
    hashes = json.loads((AUDIT / "input_hashes.json").read_text())
    p = ROOT / "runs" / "orbit_mdc" / "mdc_m2" / "model.pth"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert hashes["runs/orbit_mdc/mdc_m2/model.pth"] == h
