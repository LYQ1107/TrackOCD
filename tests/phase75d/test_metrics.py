from src.iclr27_phase75d.retrieval_metrics import score_records


def test_retrieval_metrics_and_unsafe_flip():
    rows = [{"query_key": "q", "category": 1, "video": 1, "candidates": ["p", "n"], "positives": ["p"], "negatives": ["n"], "scores": [0.9, 0.1], "raw_scores": [0.9, 0.1]}]
    out = score_records(rows)
    assert out["queries"] == 1 and out["r1"] == 1.0 and out["unsafe_flip_count"] == 0
    rows[0]["scores"] = [0.1, 0.9]
    out = score_records(rows)
    assert out["r1"] == 0.0 and out["unsafe_flip_count"] == 1
