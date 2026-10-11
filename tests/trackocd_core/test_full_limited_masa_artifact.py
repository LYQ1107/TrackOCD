"""Verify actual completed artifacts; a green test is not scientific PASS."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import pytest

from scripts.trackocd_core.report_core_training import require_complete, terminal_provenance
from src.trackocd_v2.io import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/trackocd_core"


@pytest.fixture(scope="module")
def actual():
    return json.loads((OUT / "LIMITED_MASA_EVALUATION_RESULT.json").read_text())


def test_completed_matrix_preserves_denominators_aliases_and_real_ct_exceptions(actual):
    rep = json.loads((OUT / "TRAIN_FIRST_REPRESENTATION_HELDOUT_RESULT.json").read_text())
    policy = json.loads((OUT / "TRAIN_FIRST_POLICY_HELDOUT_RESULT.json").read_text())
    require_complete(rep, policy, actual)
    counts = Counter(r["execution_id"] for r in actual["cases"])
    assert len(counts) == 840 and sum(counts.values()) == 2040
    with (OUT / "LIMITED_MASA_CASE_ALIAS_MANIFEST.csv").open() as f:
        aliases = list(csv.DictReader(f))
    assert len(aliases) == 2040
    assert sum(r["first_logical_reference_not_independent_repetition"] == "True" for r in aliases) == 840
    for r in aliases:
        assert int(r["logical_alias_count"]) == counts[r["execution_id"]]
    representation = {"A1_SELECTED", "A2_EVIDENCE", "A2_CAPACITY_CONTROL",
                      "B0_frame_snapshot_vote", "B1_track_nearest", "B2_track_dpmeans"}
    learned = [r for r in actual["cases"] if r["method"] not in representation]
    assert len(learned) == 1800
    assert all(r["persistent"]["correct_commit_ct"] == 0 for r in learned)
    # Preserve real baseline successes, not a false all-2040-zero assertion.
    positives = Counter(r["method"] for r in actual["cases"] if r["persistent"]["correct_commit_ct"] > 0)
    assert positives == {"B1_track_nearest": 1, "B2_track_dpmeans": 2}
    for r in actual["cases"]:
        assert r["unmatched_predicted_IDs_retained"] == 302972
        assert r["persistent"]["commit_ct_denominator"] == actual["fixed_full_GT"]["reuse_opportunities"][r["order"]]
    for name in ("PER_ORDER_RESULTS.csv", "ERROR_BREAKDOWN.csv"):
        with (OUT / name).open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4260 and sum(r["stage"] == "T5" for r in rows) == 2040


def test_actual_aggregates_use_seed_then_order_means_and_actual_h(actual):
    fields = {"old_acc": "standard", "new_acc": "standard", "h_score": "standard",
              "correct_commit_ct": "persistent", "false_merge_rate": "persistent",
              "effective_commit_coverage": "persistent"}
    for aggregate in actual["aggregate"]:
        rows = [r for r in actual["cases"] if (r["method"], r["prefix"], r["coverage_target"])
                == (aggregate["method"], aggregate["prefix"], aggregate["coverage_target"])]
        by_seed = defaultdict(list)
        for r in rows:
            by_seed[r["seed"]].append(r)
        assert len(by_seed) == aggregate["training_seeds"]
        assert all(len(v) == 4 and len({r["order"] for r in v}) == 4 for v in by_seed.values())
        for key, section in fields.items():
            means = [math.fsum(r[section][key] for r in v) / len(v) for v in by_seed.values()]
            mean = math.fsum(means) / len(means)
            std = math.sqrt(math.fsum((v - mean) ** 2 for v in means) / len(means))
            assert aggregate[key + "_mean"] == pytest.approx(mean, abs=1e-13)
            assert aggregate[key + "_seed_std_ddof0"] == pytest.approx(std, abs=1e-13)
    full = next(r for r in actual["aggregate"] if r["method"] == "FULL" and r["prefix"] == 16 and r["coverage_target"] == 1.)
    assert full["old_acc_mean"] > 0 and full["new_acc_mean"] > 0 and full["h_score_mean"] == 0


def test_actual_pairs_require_both_coverage_gaps_and_disclose_conditional_bootstrap(actual):
    assert len(actual["paired_comparisons"]) == 780
    primary = 0
    for r in actual["paired_comparisons"]:
        assert r["comparable_within_tolerance"] == (
            max(r["actual_reuse_coverage_gap"], r["actual_predicted_ID_commitment_coverage_gap"]) <= .05)
        if r["prefix"] == 16:
            for correction in r["paired_error_corrections"].values():
                assert correction["net_correct_change"] == correction["errors_corrected"] - correction["new_errors_introduced"]
                assert correction["denominator"] == sum(correction[k] for k in (
                    "both_correct", "both_wrong", "errors_corrected", "new_errors_introduced"))
            if r["coverage_target"] == 1.:
                primary += 1
                bootstrap = r["conditional_video_bootstrap"]
                assert (bootstrap["samples"], bootstrap["seed"], bootstrap["videos"]) == (500, 1027, 988)
                assert bootstrap["conditional_on_frozen_global_mapping_and_predicted_stream_state"] is True
                assert bootstrap["independent_video_or_category_observations_claimed"] is False
    assert primary == 84


def test_all_private_payloads_and_terminal_provenance_match_actual_public_result(actual):
    import pyarrow.parquet as pq
    from scripts.trackocd_core.run_limited_masa_inference import completed_case, inputs

    base = OUT / "core_training/train_first_v1"
    matrix = base / "limited_masa_evaluation"
    if not (matrix / "full_inference_manifest.json").exists():
        pytest.skip("Local raw assets intentionally not published; public checks still run")
    cfg, _, _, identity = inputs()
    plan = json.loads((matrix / "plan.json").read_text())
    inference = json.loads((matrix / "full_inference_manifest.json").read_text())
    assert identity == inference["identity"] == actual["input_identity"]
    assert sha256_file(matrix / "full_inference_manifest.json") == actual["full_inference_manifest_sha256"]
    assert len(plan["jobs"]) == len(inference["records"]) == 840
    lineage = terminal_provenance(matrix, inference, actual)
    assert lineage["original_frozen_inference_registration_commits"] == ["eddbc50435e87b85b7141decfb2a5dc32290ef24"]
    freeze = json.loads((base / "model_freeze.json").read_text())
    for path, sha in freeze["protected_sha256"].items():
        assert sha256_file(ROOT / path) == sha
    for key, job in plan["jobs"].items():
        seal = completed_case(matrix, job, identity)
        assert seal is not None
        directory = matrix / "cases" / key
        assert pq.read_metadata(directory / seal["ledger"]["filename"]).num_rows == 304561
        metric = json.loads((directory / "case_metrics.json").read_text())
        flags = metric["private_flags"]
        assert sha256_file(directory / flags["filename"]) == flags["sha256"]
        assert metric["sealed_ledger_sha256"] == seal["ledger"]["sha256"]
        assert metric["evaluator_source_sha256"] == actual["source_sha256"]
        assert metric["unmatched_predicted_IDs_retained"] == 302972
        assert metric["GT_in_inference"] is metric["Val_tuning"] is metric["TAO_Test_access"] is False
        assert metric["resources"]["peak_RSS_bytes"] <= 4 * 2**30
    assert cfg["system_RAM_reserve_fraction"] == .25


def test_all_988_private_feature_shards_remain_the_same_frozen_inputs(actual):
    directory = OUT / "features/masa_limited_prefix_v1"
    if not (directory / "full_manifest.json").exists():
        pytest.skip("Local feature payloads intentionally excluded from GitHub")
    manifest = json.loads((directory / "full_manifest.json").read_text())
    assert sha256_file(directory / "full_manifest.json") == actual["input_identity"]["feature_manifest_sha256"]
    assert len(manifest["records"]) == 988
    assert sum(r["tracks"] for r in manifest["records"]) == 304561
    assert sum(r["observations"] for r in manifest["records"]) == 1294110
    for r in manifest["records"]:
        p = directory / "shards" / r["npz_filename"]
        assert p.stat().st_size == r["npz_bytes"] and sha256_file(p) == r["npz_sha256"]
        assert r["GT_or_roles_input"] is r["unmatched_tracks_removed"] is False
