from src.iclr27_phase74r.observability import build_tables


def test_null_observability_uses_unique_event_denominators():
    events = [{"event_key": f"p{i}", "kind": "positive", "fold": 0} for i in range(12)] + [{"event_key": f"n{i}", "kind": "negative", "fold": 0} for i in range(12)]
    rows, summary = build_tables(events)
    assert len(rows) == 24 * 2 * 5
    assert summary["by_fold"]["0"]["total_events"] == 24
    assert summary["by_prefix"]["16"]["source"]["joint_reliable"]["denominator"] == 24
    assert summary["by_prefix"]["16"]["source"]["joint_reliable"]["numerator"] is None
    assert all(row["no_detection"] is None for row in rows)
