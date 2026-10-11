from src.trackocd_cvpr.truncated_reference_join import truncated_join


def test_reference_missing_and_unknown_predictions_not_dropped_or_background():
    gt = {0: [('a', [0, 0, .2, .2]), ('b', [.5, .5, 1, 1])],
          1: [('a', [0, 0, .2, .2]), ('b', [.5, .5, 1, 1])]}
    pred = {0: [('x', [0, 0, .2, .2]), ('y', [.3, .3, .4, .4])],
            1: [('x', [0, 0, .2, .2]), ('y', [.3, .3, .4, .4])]}
    matches, audit = truncated_join(gt, pred)
    assert matches == {'x': 'a'}
    assert audit['GT_reference_targets'] == 2 and audit['canonical_reference_targets_matched'] == 1
    assert audit['unknown_unmatched_predicted_tracks'] == 1
    assert audit['denominator_targets_removed'] is False


def test_multi_identity_predicted_track_not_certified_by_majority():
    gt = {0: [('a', [0, 0, 1, 1])], 1: [('a', [0, 0, 1, 1])], 2: [('b', [0, 0, 1, 1])]}
    pred = {i: [('x', [0, 0, 1, 1])] for i in gt}
    matches, audit = truncated_join(gt, pred)
    assert matches == {} and audit['canonical_reference_targets_matched'] == 0
