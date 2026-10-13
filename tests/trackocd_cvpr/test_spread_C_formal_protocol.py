import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.spread_C_formal_protocol import configuration, CONFIG, EXPECTED
from src.trackocd_cvpr.spread_known_calendar import freeze_calendar
from src.trackocd_cvpr.paired_spread_known_rejection_training import paired_candidate_fit
from tests.trackocd_cvpr.test_spread_known_rejection import data

ROOT = Path(__file__).resolve().parents[2]


def test_actual_informative_pilot_qualifies_explicit_source_not_formal_execution():
    for seed in (1027, 1028, 1029):
        cfg = configuration(ROOT, seed)
        assert cfg['optimizer_steps_each'] == 200 and cfg['episodes'] == 4
        assert not cfg['full_development_automatic_start']
        assert not cfg['pilot_or_historical_checkpoint_warmstart']
    for seed in (True, 1026, 1030):
        with pytest.raises(ValueError):
            configuration(ROOT, seed)


def test_coefficient_steps_seeds_scope_and_warmstart_changes_fail_before_pilot_or_private_input(tmp_path):
    p = tmp_path/CONFIG; p.parent.mkdir(parents=True)
    for key, changed in (('repair_coefficient', .5), ('episodes', 5),
                         ('training_seeds', [1027]), ('pilot_or_historical_checkpoint_warmstart', True),
                         ('new_nonKnown_labels_unknown_as_background_or_GT_history_repair', True)):
        p.write_text(json.dumps({**EXPECTED, key:changed}))
        with pytest.raises(ValueError, match='Exact fresh'):
            configuration(tmp_path, 1027)


def test_four_complete_synthetic_episodes_actual_pair_and_snapshots_not_single_pilot():
    inputs = data(); seed = 1028; saved = []
    plan = freeze_calendar(*inputs[:2], seed, 4, 2, lambda:None)
    _, _, summary, private = paired_candidate_fit(*inputs, seed,
        episodes=4, updates_per_episode=2, checkpoint_steps=[1, 2, 4, 8],
        guard=lambda:None, progress=lambda r,c:None,
        checkpoint=lambda arm, model, optimizer, step, episode, episode_step, rng:saved.append((arm, step)),
        frozen_calendar=plan)
    assert summary['actual_shared_physical_calendar_sha256'] == plan['selected_calendar_sha256']
    assert summary['optimizer_steps_each'] == 8
    for arm in summary['arms']:
        assert [s for a, s in saved if a == arm] == [1, 2, 4, 8]
        assert len(private[arm]) == 4
        assert all(e['metadata_frames'] == 4 and e['physical_observations'] == 3
                   for e in summary['episodes'][arm])


def test_formal_driver_closes_pilot_and_formal_sources_without_changing_qualified_numeric_path():
    from scripts.trackocd_cvpr.fit_spread_known_fixed200 import SOURCES, SCRIPT
    from scripts.trackocd_cvpr.pilot_spread_known_rejection import SOURCES as pilot_sources
    from src.trackocd_cvpr.spread_C_pilot_qualification import HASHES
    assert set(pilot_sources) <= set(SOURCES) and set(HASHES) <= set(SOURCES)
    assert all((ROOT/p).is_file() for p in SOURCES)
    text = (ROOT/SCRIPT).read_text()
    assert 'weights_only=True' in text and 'frozen_calendar=frozen' in text
    assert 'if out.exists() or terminal.exists()' in text and 'if busy()' in text
