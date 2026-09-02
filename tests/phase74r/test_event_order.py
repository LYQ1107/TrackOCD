import json
from pathlib import Path

from src.iclr27_phase74r.event_order import event_order_contract, join_evaluator_metadata, load_actual_model_event_stream


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_fallback_order_is_sorted_and_metadata_is_not_reordered(tmp_path):
    _write(tmp_path / "data/iclr27_phase19r/sources/positive_events.jsonl", [{"event_key": "pos:b", "source_tracklet_keys": [], "target_tracklet_key": "", "source_video": 1, "target_video": 2}, {"event_key": "pos:a", "source_tracklet_keys": [], "target_tracklet_key": "", "source_video": 1, "target_video": 2}])
    _write(tmp_path / "data/iclr27_phase19r/sources/negative_events.jsonl", [{"event_key": "neg:b", "source_tracklet_keys": [], "target_tracklet_key": "", "source_video": 1, "target_video": 2}])
    rows, provenance = load_actual_model_event_stream(tmp_path)
    assert [x["event_key"] for x in rows] == ["neg:b", "pos:a", "pos:b"]
    metadata = [{"event_key": "meta:z", "polarity": "positive", "fold": 0}, {"event_key": "meta:a", "polarity": "negative", "fold": 1}]
    joined = join_evaluator_metadata(rows, metadata)
    contract = event_order_contract(rows, provenance, metadata, joined)
    assert contract["join_preserves_model_subsequence"]
    assert contract["join_preserves_metadata_denominator"]
    assert contract["metadata_count"] == 2


def test_public_manifest_line_order_is_authoritative(tmp_path):
    _write(tmp_path / "outputs/iclr27_phase19r/manifests/public_model_events.jsonl", [{"event_key": "z", "source_tracklet_keys": [], "target_tracklet_key": "", "source_video": 1, "target_video": 2}, {"event_key": "a", "source_tracklet_keys": [], "target_tracklet_key": "", "source_video": 1, "target_video": 2}])
    rows, provenance = load_actual_model_event_stream(tmp_path)
    assert provenance["manifest_exists"]
    assert [x["event_key"] for x in rows] == ["z", "a"]
