import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from src.trackocd_cvpr.raw_membership_development import (
    FrozenMembershipScorer, readonly_membership_stream, external_group_target,
    conditional_diagnostics, full_development_reliability)
from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit, parameter_sha256
from src.trackocd_cvpr.raw_membership_development_protocol import configuration, CONFIG, EXPECTED
from src.trackocd_cvpr.C_development_rollout import readonly_C_stream
from tests.trackocd_cvpr.test_C_development_rollout import inputs, video

ROOT = Path(__file__).resolve().parents[2]


def head():
    return NamedKnownLogit().eval().requires_grad_(False)


def fixture():
    videos, labels, known, bindings, cfg = inputs()
    cfg['K_per_family']['Top8_conditioned'] = 8
    return videos, labels, known, bindings, cfg


def test_adapter_initial_actual_controller_events_equal_frozen_Raw_all_orders_and_ids():
    videos, labels, known, bindings, cfg = fixture(); model = head(); before = parameter_sha256(model)
    for order in (1027, 1028, 1029):
        raw, rp = readonly_C_stream(videos, labels, known, bindings, cfg, 'Raw_Top8', order, lambda:None)
        fitted, fp = readonly_membership_stream(videos, labels, known, bindings, cfg, order, lambda:None, head=model)
        assert fitted == raw and [r.events for _, r in fp] == [r.events for _, r in rp]
        renamed, xp = readonly_membership_stream(videos, labels, known, bindings, cfg, order,
            lambda:None, head=model, rename_and_reverse=True)
        assert renamed == fitted and [r.events for _, r in xp] == [r.events for _, r in fp]
    assert parameter_sha256(model) == before
    assert sum(p.numel() for p in FrozenMembershipScorer(model).parameters()) == 2


def test_actual_future_visual_mutation_keeps_nonempty_sealed_earlier_events():
    videos, labels, known, bindings, cfg = fixture()
    a = {k:v.copy() for k,v in videos[0][1].items()}
    a['visual'] = np.repeat(a['visual'], 2, axis=0); a['geometry'] = np.repeat(a['geometry'], 2, axis=0)
    a['quality'] = np.ones(2, np.float32); a['offsets'] = np.array([0, 2], np.int64)
    a['frame_index'] = np.array([1, 3], np.int64); a['clock_frame_index'] = np.array([0, 1, 2, 3], np.int64)
    videos = [(videos[0][0], a), *videos[1:]]; cfg.update(observations=4, metadata_frames=8, empty_frames=4)
    changed = {k:v.copy() for k,v in a.items()}; changed['visual'][1] = 0; changed['visual'][1, 6] = 1
    model = head(); model.affine.data.copy_(torch.tensor([1.2, -.5], dtype=torch.float64))
    run = lambda data:readonly_membership_stream(data, labels, known, bindings, cfg, 1027, lambda:None, head=model)
    measured, original = run(videos); _, altered = run([(videos[0][0], changed), *videos[1:]])
    earlier = lambda rows:[r.events for v,r in rows if v == videos[0][0] and r.frame_index <= 1]
    assert any(earlier(original)) and earlier(original) == earlier(altered)
    assert (measured['observations'], measured['metadata_frames'], measured['empty_frames']) == (4, 8, 4)


def test_complete_reliability_retains_unknown_short_empty_and_all_prefix_denominators():
    videos, labels, known, _, cfg = fixture()
    videos.append((4, video(40, 9))); labels[(4, '40')] = ('no_retained_semantic_truth', None)
    cfg.update(physical_tracks=4, observations=4, metadata_frames=8, empty_frames=4)
    model = head(); before = parameter_sha256(model)
    r = full_development_reliability(videos, labels, known, cfg, lambda:None, head=model, initial=head())
    assert r['full_clock'] == {'metadata_frames':8, 'empty_frames':4, 'physical_observations':4,
        'canonical_groups':4, 'Known_groups':1, 'declared_pseudoNovel_groups':2, 'masked_groups':1}
    for p in r['fixed_observed_prefix_track_tasks'].values():
        assert p['physical_tracks'] == 4 and p['masked_tracks'] == 1
        for k in ('initial_fixed_Raw_task_score', 'fixed_formal_endpoint_task_score'):
            assert p[k]['lawful_groups'] == 3 and p[k]['Known_groups'] == 1 and p[k]['declared_pseudo_groups'] == 2
            assert sum(b['lawful_group_count'] for b in p[k]['bins']) == 3
        assert p['initial_fixed_Raw_task_score'] == p['fixed_formal_endpoint_task_score']
    assert not r['unknown_conflicting_groups_as_negative_BG_Novel']
    assert not r['private_score_target_tables_exported'] and parameter_sha256(model) == before
    with pytest.raises(ValueError, match='Full physical denominator'):
        full_development_reliability(videos[:-1], labels, known, cfg, lambda:None, head=model, initial=head())


def test_conflicting_aliases_and_outside_are_masked_not_split_or_negative():
    labels = {(1, 'a'):('Known', 7), (1, 'b'):('declared_pseudoNovel', 9),
              (1, 'c'):('outside_eligible_Known', None), (1, 'd'):('Known', 7)}
    assert external_group_target(labels, 1, ('a', 'b')) is None
    assert external_group_target(labels, 1, ('c',)) is None
    assert external_group_target(labels, 1, ('a', 'd')) == ('Known', 7)
    labels[(1, 'c')] = ('outside_eligible_Known', 11)
    with pytest.raises(ValueError, match='Masked'):external_group_target(labels, 1, ('c',))


def test_binary_task_diagnostics_exact_NLL_Brier_bins_not_claimed_deployment_calibration():
    r = conditional_diagnostics(head(), {('Known', 7):[0.], ('declared_pseudoNovel', 9):[0.]})
    p = 1/(1+np.exp(.65))
    assert r['micro_NLL'] == pytest.approx((np.logaddexp(0, -.65)*2+.65)/2)
    assert r['micro_Brier'] == pytest.approx(((p-1)**2+p**2)/2)
    assert r['fixed_10_bin_micro_ECE'] == pytest.approx(abs(p-.5))
    assert not r['calibration_quality_proved_or_model_selected_from_diagnostics']
    with pytest.raises(ValueError, match='Both independent'):conditional_diagnostics(head(), {('Known', 7):[.5]})


def test_no_external_truth_feedback_to_causal_event_scores():
    videos, labels, known, bindings, cfg = fixture(); model = head()
    run = lambda truth:readonly_membership_stream(videos, truth, known, bindings, cfg, 1027, lambda:None, head=model)
    _, first = run(labels)
    alternate = dict(labels); alternate[(2, '20')] = ('declared_pseudoNovel', 10)
    _, second = run(alternate)
    assert [r.events for _, r in first] == [r.events for _, r in second]


def test_frozen_requirement_and_original_candidate_interface():
    with pytest.raises(ValueError, match='Frozen'):FrozenMembershipScorer(NamedKnownLogit())
    args = fixture(); args[-1]['K_per_family']['Top8_conditioned'] = 1
    with pytest.raises(ValueError, match='Same fixed'):readonly_membership_stream(*args, 1027, lambda:None, head=head())


def test_actual_all4_config_no_private_snapshots_or_dev_labels_opened():
    for control, seed in (('calendar_matched', 1027), ('calendar_matched', 1028), ('calendar_matched', 1029), ('full_policy', None)):
        cfg = configuration(ROOT, control, seed)
        assert (cfg['physical_tracks'], cfg['observations'], cfg['metadata_frames'], cfg['empty_frames']) == (472, 8753, 2562, 116)
        assert cfg['optimizer_steps'] == 0 and not cfg['select_model_or_checkpoint']
        assert cfg['all13_private_snapshot_fingerprints_required_before_dev']
        assert cfg['full3orders_and_nonvacuous_ID_reverse_future_probes_required']


@pytest.mark.parametrize('key,value', [
    ('order_seeds', [1027]), ('physical_tracks', 421), ('observations', 100),
    ('calendar_checkpoint_step', 50), ('optimizer_steps', 1), ('select_model_or_checkpoint', True),
    ('conditional_reliability_bins', 20), ('fixed_existing_gate', .6),
    ('all13_private_snapshot_fingerprints_required_before_dev', False),
    ('full3orders_and_nonvacuous_ID_reverse_future_probes_required', False),
    ('wall_seconds_ceiling', 6000), ('maximum_attempts_per_control_seed', 2),
])
def test_protocol_relaxations_fail_before_any_real_private_or_dev_input(tmp_path, key, value):
    p = tmp_path/CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps({**EXPECTED, key:value}))
    with pytest.raises(ValueError, match='Exact ALL4'):configuration(tmp_path, 'calendar_matched', 1027)


@pytest.mark.parametrize('control,seed', [('full_policy', 1027), ('calendar_matched', None), ('calendar_matched', True)])
def test_no_fake_fullpolicy_seeds_or_missing_calendar_seed(control, seed):
    with pytest.raises(ValueError, match='Exact ALL4'):configuration(ROOT, control, seed)


def test_supervisor_forwards_explicit_control_seed_and_requires_complete_result(tmp_path, monkeypatch):
    import scripts.trackocd_cvpr.evaluate_raw_membership_development as driver
    commands = []
    class Child:
        def poll(self): return 0
        def wait(self, timeout): return 0
    monkeypatch.setattr(driver, 'ROOT', tmp_path); monkeypatch.setattr(driver, 'busy', lambda:False)
    monkeypatch.setattr(driver, 'source_gate', lambda *a:None)
    monkeypatch.setattr(driver.runtime, 'resources_ok', lambda *a:None)
    monkeypatch.setattr(driver.subprocess, 'Popen', lambda cmd, **kw:(commands.append(cmd), Child())[1])
    cfg = {'output_directory':'new_dev', 'rss_bytes_ceiling':1, 'output_bytes_ceiling':9999}
    args = SimpleNamespace(control='calendar_matched', seed=1029, preregistration_commit='source', git_proxy='')
    assert driver.supervise(args, cfg, {'system_ram_reserve_fraction':0.}) == 1
    assert commands[0][commands[0].index('--seed')+1] == '1029'
    receipt = json.loads((tmp_path/'new_dev_supervisor_attempt_1.json').read_text())
    assert not receipt['successful_owned_execution'] and not receipt['complete_result_present']
    with pytest.raises(ValueError, match='One attempt'):driver.supervise(args, cfg, {'system_ram_reserve_fraction':0.})


def test_source_closure_launcher_preserves_actual_formal_numerical_dependencies_and_explicit_seed():
    from scripts.trackocd_cvpr.evaluate_raw_membership_development import SOURCES
    from scripts.trackocd_cvpr.fit_raw_known_membership_formal import SOURCES as FIT_SOURCES
    from scripts.trackocd_cvpr.launch_owned_raw_membership_development import supervisor_command
    assert set(FIT_SOURCES) <= set(SOURCES) and all((ROOT/p).is_file() for p in SOURCES)
    for control, seed in (('calendar_matched', 1028), ('full_policy', None)):
        cmd = supervisor_command(SimpleNamespace(control=control, seed=seed, preregistration_commit='f'*40, git_proxy='loopback'))
        assert cmd[cmd.index('--control')+1] == control and '--worker' not in cmd
        assert ('--seed' in cmd) == (seed is not None)
