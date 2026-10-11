"""Actual broad completion audit; scientific failure is not engineering failure."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from src.trackocd_v2.io import sha256_file
from scripts.trackocd_core.enrich_final_handoff import per_seed_four_orders

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/trackocd_core'


def test_actual_complete_handoff_binds_every_deliverable_without_promoting_m1():
    d=json.loads((OUT/'FINAL_RESULT.json').read_text())
    assert d['status']=='COMPLETE_AMENDED_RESEARCH_NEGATIVE_NOT_STRONG_END_TO_END_SUCCESS'
    assert d['original_M0_M10_all_requirements_audited'] and not d['strong_end_to_end_success']
    assert [r['stage'] for r in d['required_stage_dispositions']]==[f'M{i}' for i in range(11)]
    assert d['required_stage_dispositions'][1]['scientific_status']=='BLOCKED'
    assert d['historical_M1_status']=='BLOCKED_FRONTEND_QUALITY' and d['PHE']['status']=='INCOMPARABLE'
    assert d['M11']=='NOT_AUTHORIZED_INDEPENDENT_SCIENTIFIC_GATES_FAILED'
    assert (d['actual_fits'],d['retained_checkpoints'],d['controlled_cases'],d['unique_Val_executions'],d['logical_Val_cases_including_aliases'])==(33,99,2220,840,2040)
    for group in ('artifact_sha256','prior_evidence_sha256'):
        for path,sha in d[group].items():assert sha256_file(ROOT/path)==sha
    for dest,src in d['explicit_byte_aliases'].items():assert (OUT/dest).read_bytes()==(OUT/src).read_bytes()
    for key in ('new_training','new_physical_inference','new_GT_join','Val_training_or_tuning','TAO_Test_access','foreign_process_interference','new_asset_downloads'):assert d[key] is False
    assert d['canonical_handoff_enrichment']['actual_per_seed_four_order_rows']==510
    from PIL import Image
    for name in ('FINAL_PREFIX_CURVES.png','FINAL_COVERAGE_RISK_POINTS.png'):
        with Image.open(OUT/name) as im:assert im.width>=1600 and im.height>=700;im.verify()
    text=(ROOT/'docs/trackocd_core/FINAL_REPORT.md').read_text()
    assert 'NOT the primary result' in text and 'not strong end-to-end' in text
    assert 'not a hard ceiling' in text and 'not all 2040 cases have zero CT' in text
    assert 'PER_SEED_FOUR_ORDER_RESULTS.csv' in text and 'unresolved reuse WAIT/missing' in text


def test_actual_510_seed_statistics_and_165_ablation_rows_match_completed_inputs():
    val=json.loads((OUT/'LIMITED_MASA_EVALUATION_RESULT.json').read_text())
    with (OUT/'PER_SEED_FOUR_ORDER_RESULTS.csv').open() as f:rows=list(csv.DictReader(f))
    expected=per_seed_four_orders(val);assert len(rows)==len(expected)==510
    for r,e in zip(rows,expected):
        assert r['method']==e['method'] and int(r['prefix'])==e['prefix']
        for k,v in e.items():
            if isinstance(v,float):assert float(r[k])==pytest.approx(v,abs=1e-14)
    with (OUT/'ABLATION_RESULTS.csv').open() as f:ablation=list(csv.DictReader(f))
    assert len(ablation)==165
    assert not any(r['method'].startswith('B0_') or r['method'].startswith('B1_') or r['method'].startswith('B2_') for r in ablation)
    assert sum(r['method']=='FULL' for r in ablation)==15


def test_actual_v2_recovery_and_all_checkpoints_are_preserved_without_reset():
    recovery=json.loads((OUT/'audit/source_recovery.json').read_text())
    assert recovery['file_count']==len(recovery['files'])==81 and recovery['nas_git_a100_blob_identity_verified']
    for r in recovery['files']:
        content=subprocess.check_output(['git','show',recovery['recovery_commit']+':'+r['path']],cwd=ROOT)
        assert hashlib.sha256(content).hexdigest()==r['sha256'] and len(content)==r['bytes']
    core=json.loads((OUT/'CORE_TRAINING_FINAL_RESULT.json').read_text())
    for cp in core['retained_checkpoints']:
        p=Path(cp['path']);assert p.stat().st_size==cp['bytes'] and sha256_file(p)==cp['sha256']
    for path,sha in core['protected_sha256'].items():assert sha256_file(ROOT/path)==sha
    main=subprocess.check_output(['git','rev-parse','refs/heads/main'],cwd=ROOT,text=True).strip()
    assert main=='7b1d3c09f2553450a078a7cee70518a18ef60613'
    stashes=subprocess.check_output(['git','stash','list','--format=%H'],cwd=ROOT,text=True).splitlines()
    assert {'13f322b946a43644516917580fd02548d132dcac','0851687c861c5048ac4ce3944f8cd447d88bfcf3'}<=set(stashes)
