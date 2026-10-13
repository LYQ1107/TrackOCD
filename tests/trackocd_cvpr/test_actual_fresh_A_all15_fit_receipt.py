"""Actual full fifteen-fit public cohort, not development or Goal acceptance."""
import json
from pathlib import Path
from src.trackocd_cvpr.fresh_A_formal_fit import verified_cohort, PREFIX
from src.trackocd_cvpr.fresh_A_baseline import FAMILIES

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = 'outputs/trackocd_cvpr/cvpr2/fresh_A_fixed200_all15_fit_receipt_v1.json'


def test_actual_full15_receipt_matches_complete_public_cohort_without_weight_reads():
    report = json.loads((ROOT / RECEIPT).read_text())
    cohort = verified_cohort(ROOT, check_private_endpoints=False)
    assert report['cohort'] == cohort
    assert len(cohort['trials']) == 15 and cohort['all15_actual_complete_fixed_fits']
    assert cohort['same_seed_initialization_and_calendar_verified']
    assert cohort['three_distinct_actual_seeds_verified']
    assert not cohort['private_endpoint_hashes_checked'] and not cohort['full_dev_or_Goal_accepted']
    assert not report['dev_or_Goal_accepted']
    totals, peak, checkpoint_bytes = 0., 0, 0
    for row in cohort['trials']:
        out = ROOT / '{}_{}_s{}'.format(PREFIX, row['family'], row['seed'])
        terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
        result = json.loads((out / 'result.json').read_text())
        totals += terminal['wall_seconds_including_worker_imports_and_network']
        peak = max(peak, terminal['peak_sampled_owned_worker_RSS_bytes'])
        checkpoint_bytes += sum(c['bytes'] for c in result['private_checkpoints'])
    assert report['resources'] == {
        'sum_owned_wall_seconds': totals, 'maximum_sampled_RSS_bytes': peak,
        'training_GPU_hours': 0, 'private_checkpoints_bytes': checkpoint_bytes,
    }
    expected = {'1027': (498, 226), '1028': (495, 221), '1029': (498, 223)}
    for seed, values in report['actual_exposure_per_seed'].items():
        assert [r['family'] for r in values] == list(FAMILIES)
        assert {(r['groups'], r['videos']) for r in values} == {expected[seed]}
        assert expected[seed][0] < 1158  # Eligible census is not all-group exposure.
