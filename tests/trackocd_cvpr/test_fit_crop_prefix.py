import json
import numpy as np
from src.trackocd_cvpr.fit_crop_prefix import fit_physical_tracks, select_fit_prefix, check_actual_prefixes


def test_fit_gate_skips_other_geometry_and_track_ID_values(tmp_path):
    p = tmp_path / 'fixture.json'
    obs = [[1, 0, 1, 2, 3, 4]]
    p.write_text(json.dumps({'tracks': [{'video_id': 10, 'opaque_track_id': {'do_not_decode': True}, 'observations': obs},
        {'observations': {'protected': 'not_a_list'}, 'video_id': 99}]}))
    assert fit_physical_tracks(p, {10}) == [{'video_id': 10, 'observations': obs}]


def test_singleton_selection_does_not_filter_or_force_two_observations():
    image = dict(id=1, video_id=10, frame_index=0, width=20, height=10, file_name='train/a.jpg')
    tasks = select_fit_prefix({'image_metadata': [image]}, [{'video_id': 10, 'observations': [[1, 0, 1, 2, 3, 4]]}])
    assert len(tasks) == 1 and tasks[0]['GT_control_xyxy'] == [1, 2, 4, 6]


def test_numeric_R_prefix_preservation_and_future_mutation_without_fit():
    visual = np.zeros((2, 768), np.float32); visual[0, 0] = 1; visual[1, 1] = 1
    rows = [dict(width=10, height=10, frame_index=f, GT_control_xyxy=[-1, 0, 5, 5]) for f in (3, 7)]
    check = check_actual_prefixes(visual, rows)
    assert check['actual_observation_prefixes_checked'] == [1, 2]
    assert check['R_identity_start_Raw_exact'] and check['future_tail_mutation_prefix_exact']
    assert check['R_parameters_untrained_unchanged'] and check['not_a_frame_online_controller_or_trained_method']
