import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/R_development_first_pair_v1'


def test_actual_firstpair_dev_keeps_full_gallery_queries_failures_and_not_winner():
    p = OUT / 'result.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '4cfab1308f6e01843cc27824a9071501844ae9e971a8aa4dd0610ed5cee66483'
    r = json.loads(p.read_text())
    assert r['live_remote_registration_proof']['sha'] == '9f168dd67a8f68c911360500d3df3f894b529087'
    assert r['development_scope']['physical_tracks'] == 472
    assert r['development_scope']['physical_label_strata'] == dict(Known=302, declared_pseudoNovel=119,
        outside_eligible_Known=2, no_retained_semantic_truth=49)
    assert not r['model_or_checkpoint_selection'] and r['optimizer_steps'] == 0
    assert not r['final_Val_Test_or_new_nonKnown_semantic_values_read']
    assert not r['complete_semantic_truth_Hscore_purity_or_R_C_M_Goal_accepted']
    for variant, rows in r['results'].items():
        assert set(rows) == {'1', '2', '4', '8', '16', 'all_available'}
        for cap, row in rows.items():
            q = row['R']['retrieval']
            assert q['declared_query_tracks'] == 119 and q['physical_gallery_tracks'] == 472
            assert row['Raw'] == r['Raw_reference'][cap]
            assert q['not_complete_semantic_GT_purity_Hscore_or_final_acceptance']
    allcap = r['results']['temporal_Raw_teacher']['all_available']['R']['retrieval']
    assert allcap['certified_lower_bound_macro_R1'] < r['Raw_reference']['all_available']['retrieval']['certified_lower_bound_macro_R1']
    temporal = r['results']['temporal_Raw_teacher']; timeless = r['results']['timeless_Raw_teacher']
    assert all(temporal[c]['R']['retrieval'] == timeless[c]['R']['retrieval'] for c in temporal)
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
