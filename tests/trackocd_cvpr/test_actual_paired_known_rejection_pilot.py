"""Real pilot public receipts: normal plumbing, intervention never optimized."""
import json
from pathlib import Path
from src.trackocd_v2.io import sha256_file
from scripts.trackocd_cvpr.pilot_paired_known_rejection import SOURCES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/trackocd_cvpr/cvpr3/paired_Known_set_rejection_pilot50_v1'


def test_actual_normal_paired50_pilot_cannot_qualify_unexercised_repair():
    terminal=OUT.with_name(OUT.name+'_supervisor_attempt_1.json')
    assert sha256_file(OUT/'result.json')=='9a67853e3070dec7d86f7b7b0b7638e9f68b48d374950bcd490692f6a58a8239'
    assert sha256_file(OUT/'launch.json')=='94ce0efb56b31140191b2d12853fbe5b405dcfab3d6b1413903334348e506b71'
    assert sha256_file(terminal)=='1c746d836d3c9b5d7192c057d607142747736ff4876745efb1b2354114dbdc13'
    r=json.loads((OUT/'result.json').read_text()); t=json.loads(terminal.read_text())
    launch=json.loads((OUT/'launch.json').read_text()); s=r['summary']
    assert t['successful_owned_execution'] and t['owned_worker_exit_code']==0 and t['guard_stop_reason'] is None
    assert t['complete_result_present'] and t['CPU_compute_workers']==1 and t['training_GPU_hours']==0
    assert t['peak_sampled_worker_threads']<=2 and t['peak_identified_native_driver_helpers']<=1
    assert t['peak_sampled_owned_worker_RSS_bytes']<3*1024**3
    assert t['wall_seconds_including_worker_imports_network']<3000
    assert not r['resources']['CUDA_initialized']
    assert launch['preregistration_commit']=='1a5f81bc075d2e2cfca2257f83534242a8bf4559'
    assert launch['live_remote_registration_proof']==r['live_remote_registration_proof']
    assert set(launch['source_sha256'])==set(SOURCES)
    for path, expected in launch['source_sha256'].items(): assert sha256_file(ROOT/path)==expected
    assert s['optimizer_steps_each']==50 and s['trainable_parameters_each']==708
    assert len(set(s['initialization_sha256'].values()))==1
    assert len(set(s['endpoint_parameter_sha256'].values()))==1
    assert s['initialization_sha256']['baseline']!=s['endpoint_parameter_sha256']['baseline']
    for arm in s['arms']:
        rows=s['loss_curves'][arm]; episode=s['episodes'][arm][0]
        assert [row['step'] for row in rows]==list(range(1,51))
        assert sum(row['base_family_groups'].get('Known',0) for row in rows)==175
        assert sum(row['base_family_groups'].get('policy_pseudo',0) for row in rows)==0
        assert all(row['rejection_macro_loss'] is None and row['rejection_parameter_gradient_norm']==0 for row in rows)
        assert (episode['videos'],episode['metadata_frames'],episode['empty_frames'],episode['physical_observations'])==(10,374,1,1268)
        assert episode['loss_opportunities']['rejection_eligible_groups_all_frames']==206
        assert episode['loss_opportunities']['available_rejection_groups_on_updated_frames']==0
        assert episode['physical_actions']=={'KNOWN':1242,'NEW':26}
    assert all(c['step']==50 and c['reopen_exact'] for c in r['private_checkpoints']) and len(r['private_checkpoints'])==2
    assert not r['actual_nonzero_rejection_gradient']
    assert not r['pilot_informative_for_potential_later_formal_registration']
    assert not r['pilot_warmstart_formal_automatic_start_or_dev_selection']
    assert not r['dev_final_Val_Test_inputs_used']
    # Tail opportunity is NOT optimizer exposure, failure of the objective or a
    # semantic-independent sample count. Private checkpoints aren't read here.
