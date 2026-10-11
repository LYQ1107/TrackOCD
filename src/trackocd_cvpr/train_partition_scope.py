"""Administrative exclusion boundary; never a semantic training/forward API."""
from collections import Counter
from src.trackocd_cvpr.metadata_only import TRAIN_PARTITIONS


def partition_video_scope(rows, expected_tracks=None):
    if not rows or (expected_tracks is not None and len(rows) != expected_tracks):
        raise ValueError('Expected nonempty sealed-plan administrative rows')
    owners, counts = {}, Counter()
    for row in rows:
        if (set(row) != {'video_id', 'partition'} or type(row['video_id']) is not int
                or row['video_id'] < 0 or row['partition'] not in TRAIN_PARTITIONS):
            raise ValueError('Administrative video/partition fields only')
        video, part = row['video_id'], row['partition']
        if video in owners and owners[video] != part:
            raise ValueError('Video crosses old protected/authorized partition boundary')
        owners[video] = part
        counts[part] += 1
    if set(counts) != set(TRAIN_PARTITIONS):
        raise ValueError('Every old partition must be retained, not drop a conflict')
    private = {part: sorted(video for video, owner in owners.items() if owner == part) for part in TRAIN_PARTITIONS}
    public = {'selected_rows': len(rows), 'selected_videos': len(owners),
              'partitions': {part: {'selected_rows': counts[part], 'videos': len(private[part])} for part in TRAIN_PARTITIONS},
              'old_selected_plan_global_video_disjointness_verified': True,
              'protected_video_exclusion_partition': 'heldout_selection',
              'category_absence_or_complete_annotation_census_proven': False,
              'old_nested_GT_access_undone': False, 'new_protocol_or_scientific_access_cleared': False}
    return public, private
