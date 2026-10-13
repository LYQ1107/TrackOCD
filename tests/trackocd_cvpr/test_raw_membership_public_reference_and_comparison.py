import json
from pathlib import Path
import pytest
from src.trackocd_cvpr import raw_membership_baseline_reference as reference
from src.trackocd_cvpr import raw_membership_control_comparison as comparison

ROOT = Path(__file__).resolve().parents[2]


def forbid_private_stat(monkeypatch):
    original = Path.stat
    def guard(path, *args, **kwargs):
        if path.suffix in ('.pt', '.pth'):raise AssertionError('Public reference/comparison touched private weights')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', guard)


def test_genuine_historical_Raw_reference_numeric_kernel_worker_AST_and_original_commit_bound_without_GT(monkeypatch):
    forbid_private_stat(monkeypatch); r = reference.verified_raw_reference(ROOT)
    assert r['original_source_commit'] == '88b5c56a814f2be9eaf7f760bcf172f7d248f3bd'
    assert r['original_current_numeric_worker_AST_equal'] and r['all_other_actual_source_hashes_unchanged']
    assert not r['private_GT_weights_or_score_tables_loaded'] and not r['calibration_fair_latency_coverage_winner_or_Goal_accepted']


@pytest.mark.parametrize('path', list(reference.HASHES))
def test_rewritten_original_public_result_or_guard_or_launch_refuses_before_source_GT_or_weight(tmp_path, monkeypatch, path):
    for p in reference.HASHES:
        f = tmp_path/p; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes((ROOT/p).read_bytes())
    p = tmp_path/path; r = json.loads(p.read_text()); r['invented_success'] = True; p.write_text(json.dumps(r))
    forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError, match='Exact original public Raw'):reference.verified_raw_reference(tmp_path)


def test_missing_ALL4_refuses_before_historical_baseline_or_any_private_load(tmp_path, monkeypatch):
    def no_raw(root):raise AssertionError('Partial ALL4 attempted to obtain baseline winner')
    monkeypatch.setattr(comparison, 'verified_raw_reference', no_raw); forbid_private_stat(monkeypatch)
    with pytest.raises(ValueError, match='Actual complete public'):comparison.complete_comparison(tmp_path)


def test_actual_all_order_reduction_is_not_selected_seed_order_or_independence_claim():
    r = json.loads((ROOT/(reference.DIRECTORY+'/result.json')).read_text())
    means = comparison.order_means(r)
    assert means['all_available']['Known_micro_accuracy'] == pytest.approx((12+6+4)/302/3)
    assert means['all_available']['certified_cross_video_reuse_micro_lower_bound'] == pytest.approx((13+8+13)/119/3)
    assert means['all_available']['all_physical_commit_track_coverage'] == 1.
    r['orders'].pop('1029')
    with pytest.raises(ValueError, match='All3 actual orders'):comparison.order_means(r)


def test_worker_AST_compares_numerical_operations_not_line_number_or_comments():
    assert reference.function_ast('def worker(x):\n return x+1\n', 'worker') == reference.function_ast('# old note\ndef worker(x):\n return x+1 # same\n', 'worker')
    assert reference.function_ast('def worker(x):\n return x+1\n', 'worker') != reference.function_ast('def worker(x):\n return x+2\n', 'worker')
