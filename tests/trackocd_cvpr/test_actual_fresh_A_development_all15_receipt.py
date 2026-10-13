"""Actual all15/full90-case public result audit; no private data or weights."""
import copy
import json
from pathlib import Path
import pytest
from src.trackocd_v2.io import sha256_file
from src.trackocd_cvpr.fresh_A_development_summary import summarize, CAPS, SEEDS
from src.trackocd_cvpr.fresh_A_formal_fit import verified_cohort, cohort_sources
from scripts.trackocd_cvpr.evaluate_fresh_A_development import SOURCES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/fresh_A_development_all15_v1'


def actual():
    return (json.loads((OUT / 'result.json').read_text()),
            json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text()))


def test_actual_full_native15_development_and_posthoc_summary_not_joint_or_final():
    result, terminal = actual()
    assert sha256_file(OUT / 'result.json') == 'b6b4358162dc3c1384bfe8ce9d377960594b32b8039f651a9e308c0baafe8e65'
    assert sha256_file(OUT / 'launch.json') == 'aeb7e13c6cac92ecec063f251f4db8670a91caf83e7cbb4c00fe388e72e2982c'
    assert sha256_file(OUT.with_name(OUT.name + '_supervisor_attempt_1.json')) == '6512e0fb415f1394fd3b55282161a7e32c551e5f959e1eec1ff680e03a9dd972'
    assert terminal['peak_sampled_owned_worker_RSS_bytes'] < 3 * 1024**3
    assert terminal['peak_sampled_worker_threads'] == 1 and terminal['CPU_compute_workers'] == 1
    assert terminal['training_GPU_hours'] == 0
    assert terminal['wall_seconds_including_worker_imports_network'] < 3000
    launch = json.loads((OUT / 'launch.json').read_text())
    assert launch['live_remote_registration_proof'] == result['live_remote_registration_proof']
    assert result['live_remote_registration_proof']['sha'] == 'b07526d0f7611179ea4ce144dc01e7d539b4e4c9'
    assert result['cohort'] == verified_cohort(ROOT, check_private_endpoints=False)
    assert set(launch['source_sha256']) == set((*SOURCES, *cohort_sources()))
    for source, expected in launch['source_sha256'].items():
        assert sha256_file(ROOT / source) == expected
    summary = json.loads((OUT / 'posthoc_summary.json').read_text())
    expected = summarize(result, terminal)
    expected['actual_public_artifact_sha256'] = {
        'result': sha256_file(OUT / 'result.json'), 'launch': sha256_file(OUT / 'launch.json'),
        'terminal': sha256_file(OUT.with_name(OUT.name + '_supervisor_attempt_1.json')),
    }
    assert summary == expected
    assert summary['evaluated_model_prefix_cases'] == 90 and summary['actual_fit_count'] == 15
    assert summary['Raw_reference_is_one_zero_fit_control_not_three_training_seeds']
    raw = summary['Raw_reference']['all_available']
    assert raw['R1_macro'] == pytest.approx(0.6692050585908345)
    assert raw['Known_micro'] == pytest.approx(0.9337748344370861)
    # Observed negative evidence, not manufactured acceptance or a selector.
    for family in summary['families'].values():
        assert family['all_available']['R1_macro']['max'] < raw['R1_macro']
        assert family['all_available']['R5_macro']['max'] < raw['R5_macro']
    assert summary['paired_A2_dynamic_minus_static']['all_available']['R1_macro']['max'] < 0
    assert not summary['model_or_checkpoint_selected'] and not summary['full_joint_Pred_or_Goal_accepted']


@pytest.mark.parametrize('mutation', ['missing_family', 'missing_seed', 'missing_prefix',
    'raw_changed', 'query_subset', 'future_noop', 'selector', 'bad_terminal'])
def test_public_posthoc_aggregation_cannot_salvage_incomplete_or_changed_scope(mutation):
    original, original_terminal = actual()
    result, terminal = copy.deepcopy(original), copy.deepcopy(original_terminal)
    family = result['trials']['A1_ADAPTER']
    case = family[SEEDS[0]]['cases'][CAPS[0]]
    if mutation == 'missing_family': del result['trials']['A2_EVIDENCE']
    elif mutation == 'missing_seed': del family[SEEDS[0]]
    elif mutation == 'missing_prefix': del family[SEEDS[0]]['cases'][CAPS[0]]
    elif mutation == 'raw_changed': case['Raw']['effective_rank_full_physical'] += 1
    elif mutation == 'query_subset': case['fresh_A']['retrieval']['declared_query_tracks'] = 10
    elif mutation == 'future_noop': result['numeric_probes']['A1_ADAPTER'][SEEDS[0]]['future_rows_mutated'] = 0
    elif mutation == 'selector': result['model_or_checkpoint_selected'] = True
    elif mutation == 'bad_terminal': terminal['owned_worker_exit_code'] = -9
    with pytest.raises(ValueError): summarize(result, terminal)
