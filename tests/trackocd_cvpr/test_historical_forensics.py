import pytest
import json
import numpy as np
from src.trackocd_cvpr.historical_forensics import SealedForensics


def test_unknown_not_proven_pollution_but_not_pure():
    a = SealedForensics({(1, 2): ("novel", 7)}, 2)
    a.consume(1, 1, 1, -1, 0)
    a.consume(1, 2, 2, -1, 0)
    r = a.finish()
    assert r["counts"]["proven_contamination_write_events"] == 0
    assert r["states_with_unknown_members"] == 1
    assert r["first_bad_write_sequence"] is None
    assert r["counts"]["unverified_existing"] == 1


def test_absorbing_contamination_and_duplicate_birth():
    a = SealedForensics({(1, 1): ("novel", 7), (2, 1): ("novel", 8),
                         (3, 1): ("novel", 7), (4, 1): ("novel", 7)}, 4)
    for row in [(1, 1, 1, -1, 0), (2, 1, 2, -1, 0),
                (3, 1, 2, -1, 0), (4, 1, 1, -1, 1)]:
        a.consume(*row)
    r = a.finish()
    assert r["first_proven_contamination_sequence"] == 1
    assert r["counts"]["reuse_after_proven_contamination"] == 1
    assert r["counts"]["duplicate_birth"] == 1
    assert r["mixed_states"] == 1


@pytest.mark.parametrize("rows", [[(1, 1, 2, -1, 0)],
                                   [(1, 1, 1, -1, 0), (1, 2, 1, -1, 0)]])
def test_invalid_memory_sequence_rejected(rows):
    a = SealedForensics({}, len(rows))
    with pytest.raises(ValueError):
        for row in rows:
            a.consume(*row)


def test_wrong_known_and_missing_events():
    a = SealedForensics({(1, 1): ("known", 7)}, 2)
    a.consume(1, 1, 0, 8, -1)
    assert a.counts["wrong_known"] == 1
    with pytest.raises(ValueError, match="Truncated"):
        a.finish()


def test_future_append_keeps_observed_counts_identical():
    targets = {(1, 1): ("novel", 7), (2, 1): ("novel", 8)}
    a, b = SealedForensics(targets, 1), SealedForensics(targets, 2)
    for obj in (a, b):
        obj.consume(1, 1, 1, -1, 0)
    assert a.counts == b.counts
    b.consume(2, 1, 2, -1, 0)
    assert a.finish()["mixed_states"] == 0
    assert b.finish()["mixed_states"] == 1


def test_parquet_numpy_scalars_serialize_to_strict_json():
    a = SealedForensics({(1, 1): ("known", 7)}, 1)
    a.consume(*map(np.int64, (1, 1, 0, 7, -1)))
    result = json.loads(json.dumps(a.finish(), allow_nan=False))
    assert result['counts']['correct_known'] == 1
