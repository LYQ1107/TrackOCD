from src.iclr27_phase74.q0_lineage_exporter import five_field_key, parse_five_field_key


def test_five_field_roundtrip():
    key = five_field_key(3, 7, 2, 11, 19)
    assert parse_five_field_key(key) == ("3", "7", "2", "11", "19")


def test_null_lineage_not_fabricated():
    assert five_field_key(3, None, None, 11, 19) == "3:None:None:11:19"
