from types import SimpleNamespace
import numpy as np
from src.trackocd_cvpr.known_anchor_opportunities import audit_opportunities
from tests.trackocd_cvpr.test_policy_training_data import arrays


def test_static_anchor_distinguishes_prefix_and_structural_cooccurrence():
    known={0:np.eye(1,768,0,dtype=np.float32)[0]}
    targets=SimpleNamespace(targets={(1,12):('Known',0),(1,99):('policy_pseudo',0)})
    a=audit_opportunities([(1,arrays())],targets,known,lambda:None,budget=1)
    c=a['scope_counts']
    assert c['metadata_frames']==4 and c['physical_observations']==3
    assert c['Known_eligible_frames']==2 and c['pseudo_eligible_frames']==1
    assert c['cooccurring_Known_pseudo_frames']==c['selected_pseudo_groups']==1
    assert not a['calendar_or_model_or_checkpoint_selected']


def test_disjoint_family_videos_cannot_be_fixed_by_merely_extending_known_prefix():
    known={0:np.eye(1,768,0,dtype=np.float32)[0]}
    targets=SimpleNamespace(targets={(1,12):('Known',0),(2,12):('policy_pseudo',0),(2,99):('policy_pseudo',0)})
    a=audit_opportunities([(1,arrays()),(2,arrays())],targets,known,lambda:None,budget=50)
    c=a['scope_counts']
    assert c['pseudo_groups']>0 and c['cooccurring_Known_pseudo_frames']==0
    assert c['pseudo_groups_on_all_Known_anchor_frames']==0
    assert c['masked_groups']>0 and c['metadata_frames']==8


def test_registered_audit_source_has_no_checkpoint_or_model_inference_loader():
    import scripts.trackocd_cvpr.audit_known_anchor_opportunities as driver
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    for path in driver.SOURCES: assert (root/path).is_file(),path
    source=(root/driver.SCRIPT).read_text()
    assert 'torch.load' not in source and 'original actual pilot update calendar must match'.lower() in source.lower()
