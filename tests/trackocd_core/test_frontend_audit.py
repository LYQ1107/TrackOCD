import numpy as np
import pytest

from scripts.trackocd_core.audit_frontend import audit_npz_tracks, length_statistics, recovered_nas_evidence


def _npz(path, ids):
    np.savez_compressed(path, track_id=np.array(ids), frame_index=np.array([0, 1]),
                        frame_offsets=np.array([0, len(ids) // 2, len(ids)]))


def test_id_namespacing_and_exact_observation_lengths(tmp_path):
    _npz(tmp_path / "video_1.npz", [7, 7, 9])
    _npz(tmp_path / "video_2.npz", [7, 7, 7])
    result = audit_npz_tracks(tmp_path)
    assert result["physical_tracks"] == 3
    assert result["observations"] == 6
    assert result["single_observation_fraction"] == pytest.approx(1 / 3)
    assert result["observation_length_quantiles"]["p50"] == 2
    assert not result["semantic_scores_or_prototype_ids_consumed"]


def test_non_causal_frame_order_is_not_certified(tmp_path):
    np.savez_compressed(tmp_path / "video_1.npz", track_id=np.array([7, 7]),
                        frame_index=np.array([1, 0]), frame_offsets=np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="Non-causal frame order"):
        audit_npz_tracks(tmp_path)


def test_empty_lengths_have_no_manufactured_quality_score():
    result = length_statistics(np.array([], dtype=np.int64))
    assert result["physical_tracks"] == 0
    assert result["mean_observations_per_track"] is None
import json
from pathlib import Path



def test_actual_vocabulary_evidence_supersedes_legacy_clean_declarations():
    root = Path(__file__).resolve().parents[2]
    # The public regression uses curated evidence, not private NAS payloads.
    evidence = json.loads((root / "outputs/trackocd_core/audit/nas_m1_provenance.json").read_text())
    assert evidence["qualification"]["native_legal_gate"] == "FAIL_NO_NOVEL_VOCABULARY"
    assert evidence["vocabulary"]["trackocd_novel_present_in_selected_vocab"] == 203
    assert evidence["historical_runtime_log"]["text_feature_shape"] == [296, 512]
    assert not evidence["qualification"]["primary_frontend_frozen"]


def test_metadata_identity_failure_is_not_reported_as_recovered(tmp_path):
    audit = tmp_path / "outputs/trackocd_core/audit"
    audit.mkdir(parents=True)
    payload = tmp_path / "example.json"
    payload.write_text('{}\n')
    (audit / "nas_m1_provenance.json").write_text(json.dumps({"metadata_files": [
        {"name": "example.json", "private_local_copy": "example.json", "bytes": 3, "sha256": "0" * 64}
    ]}))
    with pytest.raises(ValueError, match="metadata byte mismatch"):
        recovered_nas_evidence(tmp_path)
