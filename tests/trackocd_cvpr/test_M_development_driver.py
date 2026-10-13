import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.trackocd_cvpr import evaluate_M_development as driver
from src.trackocd_cvpr.M_development_probes import actual_M_probes
from src.trackocd_cvpr.M_development_rollout import readonly_M_stream
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from tests.trackocd_cvpr.test_M_development_prerequisites import tiny_videos

ROOT = Path(__file__).resolve().parents[2]


def args(name='Mean_latest', seed=1027):
    return SimpleNamespace(model=name, seed=seed, preregistration_commit='test-only-no-network', git_proxy='')


def fixture():
    videos = tiny_videos()
    for _, arrays in videos:
        arrays['visual'][1] = 0; arrays['visual'][1, 9] = 1
    labels = {(1, '17'): ('declared_pseudoNovel', 20), (2, '17'): ('declared_pseudoNovel', 20), (3, '17'): ('Known', 10)}
    vector = np.zeros(768, np.float32); vector[7] = 1
    cfg = {'physical_tracks': 3, 'observations': 6, 'metadata_frames': 6, 'empty_frames': 0,
        'prefix_snapshots': [1, 2, 4, 8, 16], 'max_versions': 100,
        'raw_fixed_thresholds': {'tau_known': .65, 'tau_existing': .55}}
    return videos, labels, {0: vector}, {0: 10}, cfg


@pytest.mark.parametrize('name', driver.MODELS)
def test_actual_small_M_probe_checks_earlier_events_versions_and_numeric_history(name):
    videos, labels, known, bindings, cfg = fixture()
    model = None if name not in driver.VARIANTS else MemberReliabilityMLP(pooled_control=name == 'pooled_weighted').eval().requires_grad_(False)
    first, private = readonly_M_stream(videos, labels, known, bindings, cfg, name, 1027, lambda: None, model=model)
    probe = actual_M_probes(videos, labels, known, bindings, cfg, name, 1027, lambda: None, first, private, model=model)
    assert probe['actual_full_ID_rename_reverse_stream_equal']
    assert probe['actual_future_probe'] == {'future_observations_mutated': 1,
        'verified_same_prior_events_chosen_versions_and_numeric_ledger': True}


def test_real_noop_future_probe_is_not_pass():
    videos, labels, known, bindings, cfg = fixture()
    ordinal = int(np.random.default_rng(1027).permutation(3)[0])
    arrays = videos[ordinal][1]; arrays['visual'][1] = 0; arrays['visual'][1, 0] = 1
    result, private = readonly_M_stream(videos, labels, known, bindings, cfg, 'Mean_latest', 1027, lambda: None)
    with pytest.raises(ValueError, match='no-op'):
        actual_M_probes(videos, labels, known, bindings, cfg, 'Mean_latest', 1027, lambda: None, result, private)


def test_future_probe_no_suffix_is_unavailable_not_pass():
    videos, labels, known, bindings, cfg = fixture()
    for _, arrays in videos:
        for key in ('visual', 'geometry', 'quality', 'frame_index', 'clock_frame_index'):
            arrays[key] = arrays[key][:1]
        arrays['offsets'] = np.array([0, 1], np.int64)
    cfg.update(observations=3, metadata_frames=3)
    result, private = readonly_M_stream(videos, labels, known, bindings, cfg, 'Mean_latest', 1027, lambda: None)
    with pytest.raises(ValueError, match='No actual future'):
        actual_M_probes(videos, labels, known, bindings, cfg, 'Mean_latest', 1027, lambda: None, result, private)


def test_absent_public_cohort_rejected_before_network_or_private_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    def forbidden(*a, **kw): raise AssertionError('must not reach registration or private stage')
    monkeypatch.setattr(driver, 'verify_registration', forbidden)
    monkeypatch.setattr(driver, 'verified_cohort', forbidden)
    with pytest.raises(FileNotFoundError): driver.source_gate(args(), {}, {})
    assert not list(tmp_path.iterdir())


def test_fixed_config_baseline_not_fake_fitted_seed_and_new_driver_sources_complete():
    cfg = driver.configuration(ROOT, args())
    assert cfg['output_directory'].endswith('_Mean_latest')
    assert cfg['private_assignment'].endswith('private_exact_joint_proof_or_assignment.json')
    with pytest.raises(ValueError, match='zero fits'): driver.configuration(ROOT, args(seed=1028))
    for path in ('src/trackocd_cvpr/M_development_probes.py', 'src/trackocd_cvpr/versioned_causal_outcome_audit.py',
                 'src/trackocd_cvpr/frame_atomic_state_ema.py', 'src/trackocd_cvpr/M_development_cohort.py'):
        assert path in driver.SOURCES
    assert len(driver.SOURCES) == len(set(driver.SOURCES))
    command = driver.command(args(), '--worker')
    assert command[1] == '-B' and command[2].endswith(driver.SCRIPT) and command[3] == '--worker'
    command = driver.worker_command(args(), ROOT / 'synthetic-output-not-created')
    assert command[1:3] == ['-B', '-X']
    assert command[3].endswith('synthetic-output-not-created/_fresh_no_write_cache')
    assert command[4].endswith(driver.SCRIPT) and command[5] == '--worker'


def test_manifest_or_guard_change_rejected_without_input_reads(tmp_path):
    m = json.loads((ROOT / driver.CONFIG).read_text())
    m['wall_seconds_ceiling'] += 1
    p = tmp_path / driver.CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps(m))
    p = tmp_path / m['data_config']; p.write_text('{}')
    with pytest.raises(ValueError, match='protocol'): driver.configuration(tmp_path, args())
