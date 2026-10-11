import copy
import numpy as np
import pytest
from src.trackocd_cvpr.diagnostic_feature_cache import DiagnosticFeatureCache
from src.trackocd_v2.io import sha256_file


def payload(root, video):
    (root / 'raw').mkdir(exist_ok=True)
    path = root / 'raw' / f'v{video}.npz'
    visual = np.zeros((3, 768), np.float16)
    visual[:, 0] = 1
    np.savez_compressed(path, visual=visual, geometry=np.ones((3, 4), np.float32),
        quality=np.array([.001, .5, .8], np.float32), frame_index=np.array([0, 0, 1]),
        track_id=np.array([999, 5]), offsets=np.array([0, 1, 3]))
    return {'video_id': video, 'directory': 'raw', 'npz_filename': path.name,
        'npz_bytes': path.stat().st_size, 'npz_sha256': sha256_file(path), 'tracks': 2, 'observations': 3}


def test_keeps_low_score_singletons_and_one_video_residency(tmp_path):
    records = [payload(tmp_path, v) for v in (7, 11)]
    cache = DiagnosticFeatureCache(tmp_path, records)
    assert len(cache.rows) == 4 and cache.observations == 6
    assert cache.reader is None
    view = cache.get_prefix('7:999', 16)
    assert len(view.visual) == 1 and view.quality[0] < .01
    old = cache.reader
    cache.get_prefix('11:5', 1)
    assert old.data == {} and cache.video_id == 11
    cache.close()
    assert cache.reader is None


def test_real_invisible_tail_mutation_not_assertion_only(tmp_path):
    record = payload(tmp_path, 7)
    cache = DiagnosticFeatureCache(tmp_path, [record])
    before = cache.get_prefix('7:5', 1).visual.copy()
    # Mutation occurs on the actually resident hidden-tail input.
    cache.reader.data['visual'][2] = -cache.reader.data['visual'][2]
    assert np.array_equal(before, cache.get_prefix('7:5', 1).visual)
    assert not np.array_equal(before, cache.get_prefix('7:5', 2).visual[-1:])
    cache.close()


@pytest.mark.parametrize('field,value', [('tracks', 1), ('observations', 2), ('npz_sha256', 'wrong'),
    ('npz_filename', '../v7.npz'), ('directory', '../raw')])
def test_reference_mutations_refused_not_silently_filtered(tmp_path, field, value):
    record = payload(tmp_path, 7)
    changed = copy.deepcopy(record)
    changed[field] = value
    with pytest.raises(ValueError):
        DiagnosticFeatureCache(tmp_path, [changed])


def test_duplicate_video_refused(tmp_path):
    record = payload(tmp_path, 7)
    with pytest.raises(ValueError):
        DiagnosticFeatureCache(tmp_path, [record, record])
