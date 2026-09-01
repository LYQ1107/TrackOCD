"""Focused scientific regression tests for the Phase-15R transition contract."""
from __future__ import annotations

from src.iclr27_phase15r.validation.transition_validator import (
    audit_prefix_invariance, validate_transitions,
)


def row(video, track, action, sid):
    return {"video_id": video, "track_id": track, "sem_action": action, "sem_sid": sid}


def test_novel_birth_and_same_state_cross_physical_reuse():
    rows = [row(1, 10, "new", 100000), row(1, 10, "existing", 100000),
            row(2, 20, "existing", 100000)]
    out = validate_transitions(rows, {7})
    assert out["valid"] and out["unique_new_state_count"] == 1


def test_repeated_new_is_illegal():
    out = validate_transitions([row(1, 1, "new", 4), row(1, 1, "new", 4)], {7})
    assert not out["valid"] and any(e["type"] == "repeated_new" for e in out["errors"])


def test_existing_before_new_is_illegal():
    out = validate_transitions([row(1, 1, "existing", 4)], {7})
    assert not out["valid"] and any(e["type"] == "existing_before_new" for e in out["errors"])


def test_known_supported_id():
    assert validate_transitions([row(1, 1, "known", 7)], {7})["valid"]
    assert not validate_transitions([row(1, 1, "known", 8)], {7})["valid"]


def test_prefix_truncation_no_future_dependency():
    source = [row(1, 1, "new", 100), row(1, 1, "existing", 100),
              row(2, 2, "existing", 100)]
    def replay(xs):
        return list(xs)
    assert audit_prefix_invariance(source, replay, (1, 2, 3))["valid"]


if __name__ == "__main__":
    for fn in (test_novel_birth_and_same_state_cross_physical_reuse,
               test_repeated_new_is_illegal, test_existing_before_new_is_illegal,
               test_known_supported_id, test_prefix_truncation_no_future_dependency):
        fn()
    print("phase15r transition tests: PASS")
