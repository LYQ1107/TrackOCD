import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.trackocd_cvpr.spread_C_development_protocol import configuration, CONFIG, EXPECTED
from src.trackocd_cvpr.C_development_rollout import readonly_C_stream
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
from tests.trackocd_cvpr.test_C_development_rollout import inputs

ROOT = Path(__file__).resolve().parents[2]


def test_all6_actual_completed_fit_source_gates_new_full_dev_without_loading_private_weights_or_labels():
    for arm in ('baseline', 'Known_set_rejection'):
        for seed in (1027, 1028, 1029):
            cfg = configuration(ROOT, arm, seed)
            assert (cfg['physical_tracks'], cfg['observations'], cfg['metadata_frames'], cfg['empty_frames']) == (472, 8753, 2562, 116)
            assert cfg['label_strata'] == {'Known':302, 'declared_pseudoNovel':119,
                                          'outside_eligible_Known':2, 'no_retained_semantic_truth':49}
            assert cfg['checkpoint_step'] == 200 and cfg['optimizer_steps'] == 0
            assert cfg['require_nonvacuous_ID_reverse_and_future_prefix_probes']
            assert not cfg['calibrated_Raw_or_coverage_latency_match_claimed']


@pytest.mark.parametrize('key,value', [
    ('training_seeds', [1027]), ('order_seeds', [1027]), ('physical_tracks', 421),
    ('metadata_frames', 100), ('checkpoint_step', 50), ('optimizer_steps', 1),
    ('select_model_or_checkpoint', True), ('require_nonvacuous_ID_reverse_and_future_prefix_probes', False),
])
def test_scope_or_selection_relaxation_refuses_before_any_real_cohort_private_or_dev_read(tmp_path, key, value):
    p = tmp_path/CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps({**EXPECTED, key:value}))
    with pytest.raises(ValueError, match='Exact all6'):
        configuration(tmp_path, 'baseline', 1027)


def test_same_top8_numeric_interface_complete_synthetic_orders_ID_reverse_future_and_frozen_weights():
    videos, labels, known, bindings, cfg = inputs()
    cfg['K_per_family']['Top8_conditioned'] = 8
    a = {k:v.copy() for k,v in videos[0][1].items()}
    a['visual'] = np.repeat(a['visual'], 2, axis=0)
    a['geometry'] = np.repeat(a['geometry'], 2, axis=0)
    a['quality'] = np.ones(2, np.float32); a['offsets'] = np.array([0, 2], np.int64)
    a['frame_index'] = np.array([1, 3], np.int64); a['clock_frame_index'] = np.array([0, 1, 2, 3], np.int64)
    videos = [(videos[0][0], a), *videos[1:]]
    cfg.update(observations=4, metadata_frames=8, empty_frames=4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1029); model = CandidateScoreMLP().eval().requires_grad_(False)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    def run(data, order, renamed=False):
        return readonly_C_stream(data, labels, known, bindings, cfg, 'Top8_conditioned', order,
                                 lambda:None, model=model, rename_and_reverse=renamed)
    orders = {order:run(videos, order) for order in (1027, 1028, 1029)}
    for measured, private in orders.values():
        assert (measured['observations'], measured['metadata_frames'], measured['empty_frames']) == (4, 8, 4)
        assert measured['prefix_snapshots']['all_available']['physical_tracks'] == 3
    renamed, renamed_private = run(videos, 1027, True)
    original, original_private = orders[1027]
    assert renamed == original
    assert [r.events for _,r in renamed_private] == [r.events for _,r in original_private]
    changed = {k:v.copy() for k,v in a.items()}; changed['visual'][1] = 0; changed['visual'][1,6] = 1
    _, alternate = run([(videos[0][0], changed), *videos[1:]], 1027)
    earlier = lambda rows:[r.events for video,r in rows if video == videos[0][0] and r.frame_index <= 1]
    assert any(earlier(original_private)) and earlier(original_private) == earlier(alternate)
    assert all(torch.equal(before[k], v) for k,v in model.state_dict().items())


def test_supervisor_forwards_real_arm_seed_and_requires_complete_result(tmp_path, monkeypatch):
    import scripts.trackocd_cvpr.evaluate_spread_C_development as driver
    commands = []
    class Child:
        def poll(self): return 0
        def wait(self, timeout): return 0
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'busy', lambda:False)
    monkeypatch.setattr(driver, 'source_gate', lambda *a:None)
    monkeypatch.setattr(driver.runtime, 'resources_ok', lambda *a:None)
    monkeypatch.setattr(driver.subprocess, 'Popen', lambda cmd, **kw:(commands.append(cmd), Child())[1])
    cfg = {'output_directory':'new_dev', 'rss_bytes_ceiling':1, 'output_bytes_ceiling':9999}
    args = SimpleNamespace(arm='Known_set_rejection', seed=1029, preregistration_commit='source', git_proxy='')
    assert driver.supervise(args, cfg, {'system_ram_reserve_fraction':0.}) == 1
    cmd = commands[0]
    assert cmd[cmd.index('--arm')+1] == 'Known_set_rejection' and cmd[cmd.index('--seed')+1] == '1029'
    terminal = json.loads((tmp_path/'new_dev_supervisor_attempt_1.json').read_text())
    assert not terminal['complete_result_present'] and not terminal['successful_owned_execution']


def test_persistent_command_and_source_closure_keep_original_full6_training_dependencies():
    from scripts.trackocd_cvpr.evaluate_spread_C_development import SOURCES
    from scripts.trackocd_cvpr.fit_spread_known_fixed200 import SOURCES as FIT_SOURCES
    from scripts.trackocd_cvpr.launch_owned_spread_C_development import supervisor_command
    assert set(FIT_SOURCES) <= set(SOURCES) and all((ROOT/p).is_file() for p in SOURCES)
    args = SimpleNamespace(arm='baseline', seed=1028, preregistration_commit='f'*40, git_proxy='loopback')
    cmd = supervisor_command(args)
    assert cmd[cmd.index('--arm')+1] == 'baseline' and cmd[cmd.index('--seed')+1] == '1028'
    assert '--worker' not in cmd and '--preregistration-commit' in cmd
