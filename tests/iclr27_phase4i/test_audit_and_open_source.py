"""Phase 4I open-source pinning and input-manifest tests."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_repo_commits_verified():
    inv = list(csv.DictReader(
        (ROOT / "outputs" / "iclr27_phase4i" / "open_source"
         / "repository_inventory.csv").open()))
    assert len(inv) >= 8
    for row in inv:
        name = {
            "OVTR": "OVTR", "DOVTrack": "DOVTrack", "AED": "AED",
            "EA3D": "EA3D", "open_perception": "open_perception",
            "FDTA": "FDTA", "QTrack": "QTrack", "TRACT": "TRACT",
            "OWT": "Open-World-Tracking",
        }.get(row["Method"])
        if name is None or row.get("Commit", "").startswith("n/a"):
            continue
        repo = ROOT / "third_party" / (
            name if name in ("TRACT", "Open-World-Tracking")
            else f"research_refs_phase4i/{name}")
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              check=True, capture_output=True,
                              text=True).stdout.strip()
        assert head.startswith(row["Commit"][:10]), (row["Method"], head)


def test_licenses_recorded():
    inv = list(csv.DictReader(
        (ROOT / "outputs" / "iclr27_phase4i" / "open_source"
         / "repository_inventory.csv").open()))
    assert all(row["License"] for row in inv)


def test_input_manifest():
    m = json.loads((ROOT / "outputs" / "iclr27_phase4i" / "audit"
                    / "input_manifest.json").read_text())
    assert m["subset_replay_packages"]["videos"] == 20
    assert m["subset_replay_packages"]["frames"] == 732
    assert m["subset_replay_packages"]["videos_with_track_feats"] == 20
    assert m["full_replay_packages"]["videos_with_track_feats"] == 0


def test_association_manifest_complete():
    rows = list(csv.DictReader(
        (ROOT / "outputs" / "iclr27_phase4i" / "audit"
         / "association_code_manifest.csv").open()))
    assert len(rows) >= 12
    assert any(r["component"] == "appearance_matrix" for r in rows)
    assert any(r["component"] == "motion_state" for r in rows)
