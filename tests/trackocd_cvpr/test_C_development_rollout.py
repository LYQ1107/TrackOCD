import json
from pathlib import Path
import numpy as np
import torch

from src.trackocd_cvpr.C_development_rollout import raw_fixed_scores, readonly_C_stream
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP


def video(track, axis):
    visual = np.zeros((1, 768), np.float16); visual[0, axis] = 1
    return {'track_id': np.array([track], np.int64), 'offsets': np.array([0, 1], np.int64),
        'visual': visual, 'quality': np.ones(1, np.float32), 'geometry': np.array([[0., 0., .5, .5]], np.float32),
        'frame_index': np.array([1], np.int64), 'clock_frame_index': np.array([0, 1], np.int64)}


def inputs():
    videos = [(1, video(10, 7)), (2, video(20, 0)), (3, video(30, 0))]
    labels = {(1, '10'): ('Known', 7), (2, '20'): ('declared_pseudoNovel', 9), (3, '30'): ('declared_pseudoNovel', 9)}
    cfg = {'K_per_family': {'Raw_Top1': 1, 'Raw_Top8': 8, 'Top8_pool_context': 8},
        'physical_tracks': 3, 'observations': 3, 'metadata_frames': 6, 'empty_frames': 3,
        'prefix_snapshots': [1, 2, 4], 'raw_fixed_thresholds': {'tau_known': .65, 'tau_existing': .55}}
    prior = np.zeros(768, np.float32); prior[7] = 1
    return videos, labels, {0: prior}, {0: 7}, cfg


def test_actual_Raw_K1_K8_streams_preserve_clock_denominators_and_order_not_prior_GTid():
    args = inputs()
    a, _ = readonly_C_stream(*args, 'Raw_Top1', 1027, lambda: None)
    b, _ = readonly_C_stream(*args, 'Raw_Top8', 1027, lambda: None)
    assert a == b and a['metadata_frames'] == 6 and a['empty_frames'] == 3
    assert a['prefix_snapshots']['all_available']['physical_tracks'] == 3
    assert a['prefix_snapshots']['all_available']['certified_cross_video_reuse_micro_lower_bound'] == .5
    assert a['optimizer_steps'] == 0
    renamed, private_r = readonly_C_stream(*args, 'Raw_Top8', 1027, lambda: None, rename_and_reverse=True)
    original, private_o = readonly_C_stream(*args, 'Raw_Top8', 1027, lambda: None)
    assert renamed == original
    assert [r.events for _, r in private_r] == [r.events for _, r in private_o]


def test_exact_nearest_arbitration_existing_tie_known_and_thresholds_not_margin_search():
    rows = np.zeros((2, 11), np.float32); rows[:, 0] = [.7, .7]; rows[0, 1] = 1
    scores = raw_fixed_scores(rows, np.zeros(6), tau_known=.65, tau_existing=.55)
    assert scores.argmax() == 1  # Existing wins equal Known cosine, inherited arbitration.
    rows[1, 0] = .6
    assert raw_fixed_scores(rows, np.zeros(6), tau_known=.65, tau_existing=.55).argmax() == 0


def test_readonly_frozen_C_actual_stream_no_parameter_change_or_GT_feedback():
    model = CandidateScoreMLP().eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result, private = readonly_C_stream(*inputs(), 'Top8_pool_context', 1027, lambda: None, model=model)
    assert len(private) == 6 and result['model_state_unchanged']
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_registered_full_causal_scope_and_all9_completion_gate_no_selection():
    root = Path(__file__).resolve().parents[2]
    c = json.loads((root / 'configs/trackocd_cvpr/C_GT_causal_development_v1.json').read_text())
    assert len(c['training_seeds']) * len(c['trained_variants']) == 9
    assert sum(c['label_strata'].values()) == c['physical_tracks'] == 472
    assert c['observations'] == 8753 and c['metadata_frames'] == 2562 and c['empty_frames'] == 116
    assert c['raw_fixed_thresholds'] == {'tau_known': .65, 'tau_existing': .55}
    assert c['optimizer_steps'] == 0 and not c['select_model_or_checkpoint']
    assert not c['Pred_matching_M_CT_or_Goal_accepted']


def test_missing_cohort_is_terminal_predata_gate_not_permission_to_read_dev(tmp_path):
    import pytest
    from scripts.trackocd_cvpr.evaluate_C_development import cohort_sources, verified_cohort
    paths = cohort_sources()
    assert len(paths) == 27 and len(set(paths)) == 27
    assert sum(p.endswith('/launch.json') for p in paths) == 9
    with pytest.raises(FileNotFoundError): verified_cohort(tmp_path)


def test_evaluation_supervisor_forwards_model_seed_without_defaulting_to_pilot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.trackocd_cvpr import qualify_policy_training_input as script
    commands = []
    class Child:
        def poll(self): return 0
        def wait(self, timeout): return 0
    monkeypatch.setattr(script, 'ROOT', tmp_path)
    monkeypatch.setattr(script, 'resources_ok', lambda *args: None)
    monkeypatch.setattr(script.subprocess, 'Popen', lambda cmd, **kw: (commands.append(cmd), Child())[1])
    args = SimpleNamespace(model='Top8_pool_context', seed=1029, preregistration_commit='source', git_proxy='')
    assert script.supervise(args, {'output_directory': 'new_dev', 'output_bytes_ceiling': 9999}, {},
        worker_script='scripts/trackocd_cvpr/evaluate_C_development.py') == 0
    cmd = commands[0]
    assert cmd[cmd.index('--model') + 1] == 'Top8_pool_context'
    assert cmd[cmd.index('--seed') + 1] == '1029'
