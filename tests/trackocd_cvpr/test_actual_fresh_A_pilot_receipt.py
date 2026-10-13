"""Public actual pilot evidence only, no private checkpoint or GT reads."""
import json
from pathlib import Path
from src.trackocd_v2.io import sha256_file
from src.trackocd_cvpr.fresh_A_baseline import FAMILIES
from src.trackocd_cvpr.fresh_A_training import verify_matched_pilot

ROOT=Path(__file__).resolve().parents[2]
NS='outputs/trackocd_cvpr/cvpr2/fresh_A_fit5_optimizer_pilot_v1_CPU_visibility_retry1'


def test_actual_complete_pilot_receipt_not_formal_fit_dev_or_goal_evidence():
    p=ROOT/NS; result=p/'result.json'; terminal=p.with_name(p.name+'_supervisor_attempt_1.json')
    assert sha256_file(result)=='363e5a523db3e2ecd855a0afb77642cdd6c8862a5dd35505a6b4a60ca65dd0ce'
    assert sha256_file(terminal)=='fcaed77e08b583909e270c4cbf81176d6a3727806611cca7eedb870199bc9248'
    assert sha256_file(p/'launch.json')=='fd96409e0543ec42201e1b56e9f6b123b07c70fcdcb0edf2549fd8438b7ae3b2'
    d=json.loads(result.read_text());t=json.loads(terminal.read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code']==0 and t['guard_stop_reason'] is None
    assert t['full_result_present'] and t['CPU_compute_workers']==1
    assert t['peak_sampled_worker_threads']==2 and t['peak_identified_native_driver_helpers']==1
    assert d['live_remote_registration_proof']['sha']=='28ad15966c35fe893ab99a008d5423913db5a347'
    assert d['actual_parity']==verify_matched_pilot(d['trials'])
    assert not d['actual_parity']['formal15fits_or_development_or_Goal_accepted']
    assert not d['formal_warmstart'] and not d['dev_final_Val_Test_inputs_used']
    assert not d['resources']['CUDA_initialized'] and d['resources']['training_GPU_hours']==0
    for row in d['trials']:
        assert row['scope']=={'physical_tracks':1401,'eligible_known_physical_tracks':1158,
            'masked_no_fit5_target_tracks':243,'canonical_known_groups':1158,
            'exact_same_video_Raw_aliases_deduplicated':0,'supervised_classes':5,
            'all_input_physical_tracks_accounted':True}
        assert row['actually_sampled_canonical_groups']==60 and row['actually_sampled_videos']==53
        assert row['model_parameter_sha256_before']!=row['model_parameter_sha256_after']
        assert row['adapter_loss_gradient_norm_peak']>0 and len(row['loss_curve'])==10
        assert row['batched_individual_numerically_qualified_steps']==10
        assert row['batched_individual_embedding_max_absolute_error']<1e-6
        assert row['batched_individual_parameter_gradient_max_absolute_error']<1e-6
    assert [c['family'] for c in d['private_checkpoints']]==list(FAMILIES)
    assert all(c['reopen_exact'] for c in d['private_checkpoints'])
    # Hashes are summaries, not private weight upload or independent reexecution.
