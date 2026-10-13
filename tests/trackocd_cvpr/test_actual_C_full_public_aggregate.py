import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_actual_fresh_registered_C_all9_aggregate_preserves_every_seed_order_prefix_and_Raw():
    p = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_development_all9_matrix_v1/result.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '45911586efe6112045c4233aa47a685764b2e166f286122398b29baaf9ba0e3d'
    r = json.loads(p.read_text())
    assert r['live_remote_registration_proof']['sha'] == '5db8bb3a9ef8ce57d82dd0ef95fcb8fa38f154bd'
    assert len(r['all_original_trials']) == 9
    for trial in r['all_original_trials'].values():
        assert set(trial) == {'1027', '1028', '1029'}
        for order in trial.values(): assert set(order['prefix_snapshots']) == {'1', '2', '4', '8', '16', 'all_available'}
    models = r['aggregate']['models']
    for name, known, reuse in (('Top1_conditioned', .9326710816777042, 0.),
                              ('Top8_conditioned', .6225165562913907, .018674136321195144),
                              ('Top8_pool_context', .8101545253863135, .01680672268907563)):
        end = models[name]['three_training_seed_mean']['all_available']
        assert end['Known_micro_accuracy'] == pytest.approx(known)
        assert end['certified_cross_video_reuse_micro_lower_bound'] == pytest.approx(reuse)
    assert r['zero_fit_Raw_controls']['Raw_Top1'] == r['zero_fit_Raw_controls']['Raw_Top8']
    assert not r['GT_cache_pixels_weights_read_or_optimizer_or_model_selection']
    assert not r['aggregate']['official_CT_full_semantic_truth_Pred_or_Goal_accepted']
    assert not r['aggregate']['order_replays_are_independent_training_seeds']
