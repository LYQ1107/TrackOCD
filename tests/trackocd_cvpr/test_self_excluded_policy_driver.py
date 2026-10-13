import json
from pathlib import Path
import pytest
from scripts.trackocd_cvpr.pilot_self_excluded_Mean_policy import checked_configuration, CONFIG

ROOT = Path(__file__).resolve().parents[2]


def test_registered_full_clock_orders_controls_and_limits_no_winner_or_extra_labels():
    cfg = json.loads((ROOT / CONFIG).read_text()); checked_configuration(cfg)
    for change in ({'controls': ['Mean_latest']}, {'order_seeds': [1027]}, {'observations': 100},
                   {'wall_seconds_ceiling': 6000}, {'new_optimizer_steps': 1},
                   {'scientific_self_excluded_M_EMA_full_development_active_purity_or_Goal_accepted': True}):
        with pytest.raises(ValueError): checked_configuration({**cfg, **change})


def test_real_shadow_and_actual_full_probes_include_query_views_before_acceptance():
    s = (ROOT / 'scripts/trackocd_cvpr/pilot_self_excluded_Mean_policy.py').read_text()
    assert 'old != measured or old_trace != trace' in s and 'actual_excluded_probes' in s
    assert 'verify_fingerprint_shadow(ROOT)' in s and 'runtime.verified_cohort(ROOT)' in s
    assert 'torch.load' not in s and 'return runtime.main()' in s
    assert "'7461015c0f64bf89c294630f08fac2cc8c8083a664b541c221d49bd5121f5ed5'" in s
