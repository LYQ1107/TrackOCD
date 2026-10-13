import json
from pathlib import Path
from scripts.trackocd_cvpr.qualify_frozen_raw_statistics import SOURCES, CONFIG

ROOT = Path(__file__).resolve().parents[2]


def test_numeric_qualification_has_all_actual_source_dependencies_and_no_semantic_loader():
    assert len(SOURCES) == len(set(SOURCES))
    for p in SOURCES: assert (ROOT / p).is_file()
    c = json.loads((ROOT / CONFIG).read_text())
    assert c['execution_device'] == 'CPU' and not c['qualification_or_Goal_acceptance_from_manifest']
    source = (ROOT / 'src/trackocd_cvpr/raw_statistics_qualification.py').read_text()
    assert 'project_dev_labels' not in source and 'load_completed_dev_tracks' not in source
    assert 'policy_dev_video_ids(assignment)' in source
    assert 'private_shard_index.json' in source
