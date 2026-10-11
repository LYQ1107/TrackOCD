"""Path/stat-only Train pixel coverage; no image or annotation decode."""
from collections import defaultdict
from pathlib import Path
import stat


def inventory_paths(metadata, selected_videos, frame_root, guard=lambda: None):
    root = Path(frame_root).resolve()
    train_root = (root / 'train').resolve()
    video_ids = {row['id'] for row in metadata['videos']}
    if not selected_videos or not selected_videos <= video_ids:
        raise ValueError('Selected safe videos must have Train metadata')
    by_video = defaultdict(lambda: {'annotated_images': 0, 'present_images': 0, 'present_bytes': 0,
                                    'missing_images': 0, 'invalid_existing_files': 0})
    seen_ids, seen_paths = set(), set()
    for image in metadata['images']:
        if image['video_id'] not in selected_videos:
            continue
        guard()
        relative = Path(image['file_name'])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'train':
            raise ValueError('Only safe relative Train path metadata')
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(train_root):
            raise ValueError('Train metadata symlink escapes registered Train root')
        if image['id'] in seen_ids or str(resolved) in seen_paths:
            raise ValueError('Duplicate image/path reference')
        seen_ids.add(image['id']); seen_paths.add(str(resolved))
        row = by_video[image['video_id']]
        row['annotated_images'] += 1
        try:
            info = resolved.stat()
        except FileNotFoundError:
            row['missing_images'] += 1
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            row['invalid_existing_files'] += 1
            continue
        row['present_images'] += 1
        row['present_bytes'] += info.st_size
    if set(by_video) != selected_videos:
        raise ValueError('Every selected safe video needs annotated image metadata')
    complete = {video for video, row in by_video.items() if row['present_images'] == row['annotated_images']}
    totals = {field: sum(row[field] for row in by_video.values()) for field in next(iter(by_video.values()))}
    return {'selected_safe_reference_videos': len(selected_videos), 'complete_annotated_image_videos': len(complete),
            'partial_image_videos': sum(0 < row['present_images'] < row['annotated_images'] for row in by_video.values()),
            'no_present_image_videos': sum(row['present_images'] == 0 for row in by_video.values()),
            **totals, 'pixel_bytes_hashed_or_decoded': False, 'dense_frame_coverage_certified': False,
            'missing_image_bytes_known': False, 'asset_download_or_deletion': False}, complete
