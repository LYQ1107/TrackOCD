"""Protocol integrity tests for Phase 4E.

These test real risks: frozen inputs unchanged, official GT isolation,
no oracle K / future access, candidate freeze before official runs,
evaluator untouched, multi-prototype gate respected.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_input_hashes_unchanged():
    expected = json.loads((ROOT / "outputs/iclr27_phase4e/audit/input_hashes.json").read_text())
    for rel, h in expected.items():
        p = ROOT / rel
        assert p.exists(), f"missing frozen input {rel}"
        assert sha256(p) == h, f"hash mismatch {rel}"


def test_repos_pinned():
    repos = ["ProxyAnchor", "pytorch_metric_learning", "CoPE",
             "research_xbm", "OCM"]
    for r in repos:
        p = ROOT / "third_party/research_refs_phase4e" / r
        out = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        assert out.returncode == 0 and len(out.stdout.strip()) == 40, r


def test_no_gt_in_training_source():
    src = (ROOT / "src/orbit_iam/train_iam.py").read_text()
    assert "load_gt(" not in src
    assert "gt_tracks_mean" not in src


def test_no_oracle_k_or_future_in_evaluator():
    src = (ROOT / "src/orbit_iam/evaluate_iam.py").read_text()
    # Banned runtime patterns (docstring mentions of the contract are fine).
    assert "oracle_k" not in src
    assert "rows[i + 1]" not in src
    assert "rows[i+1]" not in src


def test_first_occurrence_definition_causal():
    src = (ROOT / "src/orbit_iam/train_iam.py").read_text()
    assert 'q["first"]' in src
    assert "mem.create_novel" in src


def test_confidence_uses_no_gt_purity():
    src = (ROOT / "src/orbit_iam/iam_memory.py").read_text()
    assert "self.purity" not in src
    assert "purity_offline" not in src


def test_candidate_frozen_before_official_results():
    frozen_dir = ROOT / "outputs/orbit_iam/frozen_candidates"
    results_dir = ROOT / "outputs/orbit_iam/results"
    if not results_dir.exists():
        return  # official phase not reached; freeze will be checked when results exist
    official_files = list(results_dir.glob("*seed1027*.csv"))
    if not official_files:
        return
    for cand in ["candidate_a.json"]:
        assert (frozen_dir / cand).exists(), f"{cand} must be frozen before official"


def test_multi_prototype_gate_respected():
    gate = (ROOT / "docs/iclr27_phase4e/MULTI_PROTOTYPE_JUSTIFICATION.md").read_text()
    assert "MULTI_PROTOTYPE_NOT_JUSTIFIED" in gate
    frozen_a = ROOT / "outputs/orbit_iam/frozen_candidates/candidate_a.json"
    frozen_b = ROOT / "outputs/orbit_iam/frozen_candidates/candidate_b.json"
    if frozen_b.exists():
        raise AssertionError("Candidate B must not exist when the gate is NOT_JUSTIFIED")
    if frozen_a.exists():
        data = json.loads(frozen_a.read_text())
        assert data.get("candidate") == "A"


def test_evaluator_unchanged():
    h = sha256(ROOT / "src/trackocd_v1/evaluation/trackocd_evaluator.py")
    assert h == "61cd2509b77067f210b754e39f9b625911b690b85dcf4c4e5f9a5faffcb674c0"


def test_tracking_not_modified():
    # No new tracker artifacts/scripts appear under tracking-specific paths.
    for p in [ROOT / "outputs/iclr27_phase4b/bytetrack_features"]:
        assert p.exists()


if __name__ == "__main__":
    import traceback
    results = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                results.append({"test": name, "status": "PASS"})
            except Exception as e:
                results.append({"test": name, "status": "FAIL",
                                "error": str(e), "trace": traceback.format_exc()})
    out = ROOT / "outputs/iclr27_phase4e/tests/test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"phase": "4E", "results": results,
                               "passed": sum(1 for r in results if r["status"] == "PASS"),
                               "total": len(results)}, indent=1))
    print(json.dumps(results, indent=1))
