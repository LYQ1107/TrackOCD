from pathlib import Path
from src.iclr27_phase74.manifest_reader import read_both_preserving_order

ROOT = Path(__file__).resolve().parents[2]
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"


def test_manifest_reader_preserves_positive_line_order():
    got = list(read_both_preserving_order(POS, NEG))[:76]
    expected = [line.split('"event_key": "', 1)[1].split('"', 1)[0] for line in POS.read_text().splitlines() if line.strip()]
    assert [x.event_key for x in got] == expected


def test_manifest_reader_preserves_negative_line_order():
    got = list(read_both_preserving_order(POS, NEG))[76:]
    expected = [line.split('"event_key": "', 1)[1].split('"', 1)[0] for line in NEG.read_text().splitlines() if line.strip()]
    assert [x.event_key for x in got] == expected


def test_manifest_reader_does_not_sort_by_fold(tmp_path):
    # A deliberately out-of-fold-order fixture catches an implicit fold sort.
    p = tmp_path / "p.jsonl"; n = tmp_path / "n.jsonl"
    p.write_text('{"event_key":"p3","fold":3,"kind":"positive"}\n{"event_key":"p0","fold":0,"kind":"positive"}\n')
    n.write_text('{"event_key":"n2","fold":2,"kind":"negative"}\n')
    got = list(read_both_preserving_order(p, n))
    assert [x.event_key for x in got] == ["p3", "p0", "n2"]


def test_manifest_event_set_is_exact():
    got = list(read_both_preserving_order(POS, NEG))
    raw = [__import__('json').loads(x) for x in POS.read_text().splitlines() if x.strip()] + [__import__('json').loads(x) for x in NEG.read_text().splitlines() if x.strip()]
    assert {x.event_key for x in got} == {x['event_key'] for x in raw}


def test_manifest_no_duplicate_or_missing_event():
    got = list(read_both_preserving_order(POS, NEG))
    assert len(got) == 152 == len({x.event_key for x in got})


def test_manifest_roundtrip_preserves_json_content():
    got = list(read_both_preserving_order(POS, NEG))
    assert got[0].raw == __import__('json').loads(POS.read_text().splitlines()[0])
