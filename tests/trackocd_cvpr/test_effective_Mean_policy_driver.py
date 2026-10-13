import json
from pathlib import Path
import pytest
from scripts.trackocd_cvpr.audit_effective_Mean_policy import CONFIG, configuration

ROOT = Path(__file__).resolve().parents[2]


def test_effective_profile_keeps_entire_prior_full_policy_protocol_and_guards():
    profile = json.loads((ROOT / CONFIG).read_text()); cfg = configuration(ROOT, profile)
    old = json.loads((ROOT / profile['base_config']).read_text())
    assert {k: v for k, v in cfg.items() if k != 'output_directory'} == {k: v for k, v in old.items() if k != 'output_directory'}
    with pytest.raises(ValueError): configuration(ROOT, {**profile, 'wall_seconds_ceiling': 6000})
    with pytest.raises(ValueError): configuration(ROOT, {**profile, 'prior_result_sha256': '0' * 64})


def test_after_prediction_only_hook_requires_original_actor_full_trace_and_all18_streams():
    s = (ROOT / 'scripts/trackocd_cvpr/audit_effective_Mean_policy.py').read_text()
    assert 'len(records) != 18' in s and "current['orders'] != old['orders']" in s
    assert 'chunk[0][3:] != chunk[1][3:]' in s and 'chunk[1][3:] != chunk[4][3:]' in s
    assert "'effective_support_result.json'" in s
    o = (ROOT / 'src/trackocd_cvpr/effective_mean_observer.py').read_text()
    assert o.index('result = parent(frame, rows)') < o.index('observer.observe(' , o.index('result = parent(frame, rows)'))
