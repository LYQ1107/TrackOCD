import json
from pathlib import Path


def test_observability_denominator_artifact():
    p = Path(__file__).resolve().parents[2] / "outputs/iclr27_phase74/metrics/observability_summary.json"
    if not p.exists(): return
    d = json.loads(p.read_text())
    assert d["positive_events"] == 76 and d["negative_events"] == 76
