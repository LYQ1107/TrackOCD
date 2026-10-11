import numpy as np
from src.trackocd_cvpr.prefix_interface_checks import check_prefix_tails


def test_real_tail_mutation_and_short_track_not_discarded():
    visual = np.zeros((4, 768), np.float16)
    visual[:, 0] = 1.
    result = check_prefix_tails(visual, np.ones((4, 4), np.float32), np.ones(4, np.float32),
                                np.array([0, 1, 2, 0]), np.array([0, 3, 4]))
    assert result['prefix_checks'] == 10
    assert result['checks_with_real_tail_mutation'] == 2
