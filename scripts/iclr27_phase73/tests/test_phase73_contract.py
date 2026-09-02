from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/iclr27_phase73"

def test_phase73_status_and_null_policy_files_exist():
    assert (OUT / "export/physical_semantic_predictions.jsonl").exists()
    rows = [json.loads(x) for x in (OUT / "export/physical_semantic_predictions.jsonl").read_text().splitlines() if x.strip()]
    assert rows
    assert all(r["prediction_type"] == "unresolved" and r["action"] == "DEFER" and r["uncertainty"] == 1.0 for r in rows)

def test_event_alignment_keeps_152_events_and_prefixes():
    rows = [json.loads(x) for x in (OUT / "export/event_alignment.jsonl").read_text().splitlines() if x.strip()]
    assert len({r["event_key"] for r in rows}) == 152
    assert {r["prefix"] for r in rows} == {1, 2, 4, 8, 16}

def test_no_public_or_q1_outputs():
    forbidden = [p for p in OUT.rglob("*") if p.is_file() and any(x in p.name.lower() for x in ("q1", "devplus", "public_new"))]
    assert not forbidden
