import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.trackocd_cvpr.evaluate_C_development import recovery_configuration
from src.trackocd_cvpr.C_registration_recovery import CONFIG, PREFIX, TERMINAL

ROOT = Path(__file__).resolve().parents[2]


def test_actual_registration_failure_namespace_only_recovery_preserves_protocol_and_terminal():
    cfg = json.loads((ROOT / 'configs/trackocd_cvpr/C_GT_causal_development_v1.json').read_text())
    args = SimpleNamespace(model='Top8_pool_context', seed=1028, recovery_attempt='registration_retry1')
    result, sources = recovery_configuration(ROOT, cfg, args)
    assert result['output_prefix'] == PREFIX and result != cfg
    assert {k: v for k, v in result.items() if k != 'output_prefix'} == {k: v for k, v in cfg.items() if k != 'output_prefix'}
    assert CONFIG in sources and TERMINAL in sources
    assert not any(p.endswith('.log') for p in sources)  # raw traceback stays local
    assert not json.loads((ROOT / TERMINAL).read_text())['successful_owned_execution']


def test_registration_recovery_cannot_select_another_model_seed_or_recover_science():
    cfg = json.loads((ROOT / 'configs/trackocd_cvpr/C_GT_causal_development_v1.json').read_text())
    for model, seed in (('Top1_conditioned', 1028), ('Top8_pool_context', 1029)):
        with pytest.raises(ValueError, match='pre-input'):
            recovery_configuration(ROOT, cfg, SimpleNamespace(model=model, seed=seed, recovery_attempt='registration_retry1'))
