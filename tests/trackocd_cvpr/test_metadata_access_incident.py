"""Incident/old receipts are immutable evidence, not clean-access certificates."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
INCIDENT = ROOT / 'outputs/trackocd_cvpr/safety/train_nested_metadata_access_incident_v1.json'


def test_affected_receipts_remain_exact_and_old_reader_did_decode_whole_arrays():
    incident = json.loads(INCIDENT.read_text())
    for affected in incident['affected_direct_receipts']:
        assert hashlib.sha256((ROOT / affected['result_path']).read_bytes()).hexdigest() == affected['result_sha256']
        source = subprocess.check_output(['git', 'show', affected['registered_commit'] +
                                         ':src/trackocd_cvpr/metadata_only.py'], cwd=ROOT)
        assert hashlib.sha256(source).hexdigest() == affected['registered_reader_sha256']
        assert b"if key in ('images', 'videos'):" in source
        assert b'result[key] = json.loads(data[p:end])' in source


def test_historical_reader_exposes_synthetic_nested_gt_but_fixed_reader_does_not(tmp_path, monkeypatch):
    from src.trackocd_cvpr.metadata_only import image_video_metadata
    affected = json.loads(INCIDENT.read_text())['affected_direct_receipts'][0]
    source = subprocess.check_output(['git', 'show', affected['registered_commit'] +
                                     ':src/trackocd_cvpr/metadata_only.py'], cwd=ROOT)
    namespace = {}
    # Owned reader ONLY on synthetic fixtures; never execute an old dataset/job command.
    exec(compile(source, '<historical-owned-reader-synthetic-only>', 'exec'), namespace)
    path = tmp_path / 'synthetic.json'
    path.write_text('{"images":[],"videos":[{"id":1,"neg_category_ids":[987654321]}],'
                    '"annotations":[],"categories":[]}')
    original, decoded = json.loads, []
    def record(value, *args, **kwargs):
        decoded.append(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(json, 'loads', record)
    assert namespace['image_video_metadata'](path)['videos'][0]['neg_category_ids'] == [987654321]
    assert any(b'987654321' in value for value in decoded)
    decoded.clear()
    assert image_video_metadata(path) == {'images': [], 'videos': [{'id': 1}]}
    assert all(b'987654321' not in value for value in decoded)


def test_incident_does_not_convert_source_containment_into_scientific_clearance():
    incident = json.loads(INCIDENT.read_text())
    assert incident['status'] == 'ACCESS_SCOPE_BREACH_RECORDED_REMEDIATION_ONLY_NOT_SCIENTIFIC_CLEARANCE'
    assert incident['limits']['downstream_source_projection_does_not_undo_prior_access'] is True
    assert incident['limits']['all_historical_runtime_leakage_absent_proven'] is False
    assert incident['remediation_contract']['scientific_access_clearance_granted'] is False
    assert incident['remediation_contract']['real_metadata_rerun_performed_in_this_receipt'] is False
    assert incident['remediation_contract']['video_allowlist'] == ['id']
    assert 'train_features_v1' in incident['dependent_cvpr0_stages_access_attestation_not_revalidated']
