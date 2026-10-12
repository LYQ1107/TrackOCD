"""Fit-only GT geometric-control fixture, not a predicted frontend/cache."""
import json
import mmap
from .scoped_train_census import top_ranges, object_ranges, array_ranges, integer_value


def fit_physical_tracks(path, fit):
    rows = []
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        fields = top_ranges(data)
        for start, end in array_ranges(data, *fields['tracks']):
            spans = object_ranges(data, start, end)
            video = integer_value(data, spans['video_id'])
            if video not in fit:
                continue
            # No other-role geometry, category value, label or ID is decoded.
            observations = json.loads(data[slice(*spans['observations'])])
            if not observations or any(type(o) is not list or len(o) != 6 for o in observations):
                raise ValueError('Exact fit physical observation lists')
            rows.append({'video_id': video, 'observations': observations})
    return rows


def select_fit_prefix(plan, tracks):
    if not tracks:
        raise ValueError('No fit geometry fixture')
    # Bookkeeping video scope + visible first observation, no track ID, label,
    # difficulty/length threshold. Exact first-observation aliases remain tied;
    # stable source order is fixture selection only, not a policy tie breaker.
    track = min(tracks, key=lambda r: (r['video_id'], r['observations'][0][1], tuple(r['observations'][0][2:])))
    images = {r['id']: r for r in plan['image_metadata']}
    tasks = []
    for obs in track['observations'][:2]:
        row = images[obs[0]]
        if (row['video_id'], row['frame_index']) != (track['video_id'], obs[1]):
            raise ValueError('GT observation/image/frame source differs')
        x, y, w, h = obs[2:]
        if w <= 0 or h <= 0:
            raise ValueError('Positive GT geometry required')
        task = dict(row)
        task['GT_control_xyxy'] = [x, y, x + w, y + h]
        tasks.append(task)
    return tasks


def check_actual_prefixes(visual, rows):
    import numpy as np
    import torch
    from src.trackocd_core.features import prefix_view
    from .raw_preserving_evidence import RawPreservingEvidence
    geometry = np.asarray([np.asarray(r['GT_control_xyxy']) / [r['width'], r['height'], r['width'], r['height']]
                           for r in rows], dtype=np.float32)
    # GT-control confidence is a declared constant, NOT calibrated visibility,
    # predicted score or semantic GT purity. Unclipped amodal normalized boxes.
    quality = np.ones(len(rows), dtype=np.float32)
    frames = np.asarray([r['frame_index'] for r in rows], dtype=np.int64)
    model = RawPreservingEvidence().eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    checks = []
    for n in range(1, len(rows) + 1):
        view = prefix_view(visual, geometry, quality, frames, n)
        future_visual = np.concatenate((visual, -visual[-1:]))
        future_boxes = np.concatenate((geometry, geometry[-1:] * 2))
        future_quality = np.concatenate((quality, np.zeros(1, dtype=np.float32)))
        future_frames = np.concatenate((frames, [frames[-1] + 1]))
        future_visual[n:] *= -1
        future_boxes[n:] *= 2
        future_quality[n:] = 0
        future_frames[n:] = frames[n - 1] + np.arange(1, len(future_frames) - n + 1)
        mutated = prefix_view(future_visual, future_boxes, future_quality, future_frames, n)
        def encode(v):
            with torch.inference_mode():
                return model(torch.from_numpy(v.visual.copy()), torch.from_numpy(v.boxes_normalized_xyxy.copy()),
                             torch.from_numpy(v.quality.copy()), torch.from_numpy(v.elapsed_frames.copy()))
        a, b = encode(view), encode(mutated)
        if not torch.equal(a.embedding, a.raw) or not torch.equal(a.embedding, b.embedding):
            raise ValueError('Identity-start Raw preservation or observed-prefix future invariance failed')
        checks.append(n)
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise ValueError('Untrained evidence parameters changed')
    return {'actual_observation_prefixes_checked': checks, 'R_identity_start_Raw_exact': True,
            'future_tail_mutation_prefix_exact': True, 'R_parameters_untrained_unchanged': True,
            'GT_control_quality_constant': 1, 'GT_geometry_unclipped_amodal_normalized': True,
            'not_a_frame_online_controller_or_trained_method': True}
