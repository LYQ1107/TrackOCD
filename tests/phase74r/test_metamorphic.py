from pathlib import Path

from src.iclr27_phase74r.metamorphic import run_metamorphic


def test_real_metamorphic_fixture(tmp_path):
    result = run_metamorphic(Path(__file__).parents[2], tmp_path)
    assert all(result[key] for key in ("category_shuffle", "event_label_swap", "physical_id_renumber", "future_append", "source_before_target", "repeat_determinism", "atomic_crash", "static_antihardcode"))
