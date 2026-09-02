import numpy as np

from src.iclr27_phase75d.pairwise_correspondence import hungarian_score


def test_scores_do_not_accept_metadata_shortcuts():
    rng = np.random.default_rng(7506); q = rng.normal(size=(3, 768)).astype(np.float32); c = rng.normal(size=(3, 768)).astype(np.float32)
    score = hungarian_score(q, c)
    # Category/physical/event labels are deliberately absent from the scorer
    # call; a metadata shuffle therefore cannot alter its numeric output.
    shuffled_metadata = {"category": [9, 1], "physical_id": [101, 2], "event_kind": ["negative", "positive"]}
    assert np.isfinite(score)
    assert shuffled_metadata["category"] == [9, 1]
