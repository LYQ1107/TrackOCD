import numpy as np
import pytest
from src.trackocd_core.exact_search import ExactNearest,ExactDPMeans,ExactFrameVoting,ExactCandidateMemory,ExhaustiveMatrix
from src.trackocd_v2.methods.nearest_prototype import NearestPrototype,_normalize
from src.trackocd_v2.methods.dpmeans import OnlineDPMeans
from src.trackocd_core.baselines import FrameSnapshotNearestVoting
from src.trackocd_core.features import PrefixView
from src.trackocd_core.persistent_policy import CandidateMemory


@pytest.mark.parametrize('pair',[(NearestPrototype,ExactNearest),(OnlineDPMeans,ExactDPMeans)])
def test_many_updates_scores_ties_and_state_byte_equal(pair):
    rng=np.random.default_rng(1027);protos={17:_normalize(rng.normal(size=32)),2:_normalize(rng.normal(size=32))}
    slow,fast=[c(known_prototypes=protos) for c in pair]
    vectors=[rng.normal(size=32) for _ in range(300)]
    vectors+=vectors[:200];vectors+=list(protos.values())
    for v in vectors:assert slow.step(v)==fast.step(v)
    a=slow._anonymous if hasattr(slow,'_anonymous') else slow._centroids
    b=fast._anonymous if hasattr(fast,'_anonymous') else fast._centroids
    assert list(a)==list(b)
    for key,(v,n) in a.items():assert n==b[key][1] and np.array_equal(v,b[key][0])


def test_refinement_retains_many_exact_and_near_ties_and_distance_ties():
    rng=np.random.default_rng(1028);z=_normalize(rng.normal(size=256));m=ExhaustiveMatrix(256)
    for i in range(200):m.update(str(i),_normalize(z+rng.normal(size=256)*1e-5))
    m.update('duplicate',m.vectors[0].copy())
    for distance in (False,True):
        scores=[(k,float(1.0-np.dot(z,v)) if distance else float(np.dot(z,v))) for k,v in zip(m.ids,m.vectors)]
        scores.sort(key=lambda a:((1 if distance else -1)*a[1],m.lookup[a[0]]))
        assert m.ranked(z,k=2,distance=distance)==scores[:2]


def test_frame_pretrack_snapshot_votes_updates_and_no_provisional_writes_equal():
    rng=np.random.default_rng(1027);protos={1:_normalize(rng.normal(size=768))}
    slow=FrameSnapshotNearestVoting(known_prototypes=protos);fast=ExactFrameVoting(known_prototypes=protos)
    for i in range(100):
        v=np.stack([_normalize(rng.normal(size=768)) for _ in range(i%5+1)])
        view=PrefixView(v,np.zeros((len(v),4),np.float32),rng.uniform(.01,1,len(v)).astype(np.float32),np.arange(len(v)))
        assert slow.step_prefix(view)==fast.step_prefix(view)
    assert slow._next_id==fast._next_id
    for k,(v,n) in slow._anonymous.items():assert n==fast._anonymous[k][1] and np.array_equal(v,fast._anonymous[k][0])


def test_policy_exact_cues_and_token_lexical_ties_clear_no_recycling():
    rng=np.random.default_rng(1027);protos={17:_normalize(rng.normal(size=256)),2:_normalize(rng.normal(size=256))}
    slow=CandidateMemory(protos);fast=ExactCandidateMemory(protos)
    for i in range(150):
        z=_normalize(rng.normal(size=256));q=np.array([.7,.2],np.float32)
        a=slow.candidates(z,.2,2,q,10);b=fast.candidates(z,.2,2,q,10)
        assert np.array_equal(a[0],b[0]) and a[1:]==b[1:]
        action=2 if i<100 else 1
        assert slow.apply(action,a[1],a[2],z,q)==fast.apply(action,b[1],b[2],z,q)
    slow.clear_anonymous();fast.clear_anonymous()
    assert slow.apply(2,None,None,z,q)==fast.apply(2,None,None,z,q)=={'kind':'NEW','token':'S:100'}


def test_performance_source_has_no_supervision_and_old_baselines_unchanged():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    s=(root/'src/trackocd_core/exact_search.py').read_text()
    assert 'np.dot(vector,self.vectors[i])' in s and '8*self.dimension' in s
    assert 'evaluator_only_geometry_join' not in s and 'category_id' not in s.split('class ExhaustiveMatrix:')[1].split('class ExactNearest')[0]
