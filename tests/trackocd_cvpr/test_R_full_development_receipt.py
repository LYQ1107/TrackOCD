import hashlib
import json
from pathlib import Path
import statistics
import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/R_development_all12_v1'


def test_actual_full_R_matrix_all12_endpoints_denominators_and_numeric_probes():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '704d0e20b59e85d78a46835dd54fe53099717361f8b789e2559a5c3a95531e70'
    r = json.loads((OUT / 'result.json').read_text())
    assert len(r['results']) == 4 and r['seeds'] == [1027, 1028, 1029] and r['seed'] is None
    assert r['optimizer_steps'] == 0 and not r['model_or_checkpoint_selection']
    assert r['development_scope']['physical_tracks'] == 472
    for variant, seeds in r['results'].items():
        assert set(seeds) == {'1027', '1028', '1029'}
        for seed, rows in seeds.items():
            assert set(rows) == {'1', '2', '4', '8', '16', 'all_available'}
            for cap, row in rows.items():
                assert row['Raw'] == r['Raw_reference'][cap]
                assert row['R']['retrieval']['physical_gallery_tracks'] == 472
                assert row['R']['retrieval']['declared_query_tracks'] == 119
                assert row['R']['Known']['Known_queries'] == 302
                assert row['support']['full_physical_gallery_short_and_unlabeled_not_dropped']
            probe = r['actual_numeric_probes'][variant][seed]
            assert probe['future_rows_mutated'] == 32 and probe['actual_first_prefix_future_invariant']
            if variant.startswith('timeless'):
                assert probe['order_embedding_l2_change'] == probe['time_embedding_l2_change'] == 0
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None


def test_favorable_prefix_or_seed_does_not_prove_temporal_or_teacher_win():
    r = json.loads((OUT / 'result.json').read_text())
    end = 'all_available'; raw = r['Raw_reference'][end]['retrieval']['certified_lower_bound_macro_R1']
    ce, teacher = [], []
    for seed in ('1027', '1028', '1029'):
        ce.append(r['results']['temporal_CE'][seed][end]['R']['retrieval']['certified_lower_bound_macro_R1'])
        teacher.append(r['results']['temporal_Raw_teacher'][seed][end]['R']['retrieval']['certified_lower_bound_macro_R1'])
        for family in ('CE', 'Raw_teacher'):
            left = r['results']['temporal_' + family][seed][end]['R']['retrieval']
            right = r['results']['timeless_' + family][seed][end]['R']['retrieval']
            assert left['certified_lower_bound_macro_R1'] == right['certified_lower_bound_macro_R1']
            assert left['certified_lower_bound_micro_R1'] == right['certified_lower_bound_micro_R1']
    assert raw == pytest.approx(.6692050585908345)
    assert statistics.mean(ce) == pytest.approx(.6737645077623526)
    assert min(ce) < raw < max(ce)  # not a stable across-seed gain
    assert statistics.mean(teacher) == pytest.approx(.6573492778772951)
    assert statistics.mean(teacher) < raw
    assert not r['complete_semantic_truth_Hscore_purity_or_R_C_M_Goal_accepted']
