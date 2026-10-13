from types import SimpleNamespace
import numpy as np
import pytest
from src.trackocd_cvpr.fresh_A_development_metrics import certified_native256_retrieval,evaluate_prefix
from src.trackocd_cvpr.R_development_metrics import certified_cross_video_retrieval


def example():
    e=np.zeros((7,256));e[:,0]=1
    labels=[('declared_pseudoNovel',10),('declared_pseudoNovel',10),('Known',20),
            ('no_retained_semantic_truth',None),('declared_pseudoNovel',11),('Known',20),('Known',21)]
    return e,list(range(7)),labels


def test_native256_exact_tie_and_full_unknown_gallery_semantics_match_certified768_reference():
    e,v,l=example();raw=np.pad(e,((0,0),(0,512)))
    # Padding here is a synthetic test oracle, never actual baseline evaluation.
    a=certified_native256_retrieval(e,v,l);b=certified_cross_video_retrieval(raw,v,l)
    assert a==b and a['physical_gallery_tracks']==7 and a['declared_query_tracks']==3
    assert a['queries_without_verified_cross_video_positive_retained']==1
    assert a['queries_with_semantic_unknown_in_top_tie']==3
    assert not a['unknown_gallery_declared_background_or_labeled_Novel']


def test_native_no_samevideo_positive_invention_no_role_based_gallery_filter():
    e,v,l=example();v[1]=v[0];a=certified_native256_retrieval(e,v,l)
    assert a['queries_without_verified_cross_video_positive_retained']==3
    assert a['certified_lower_bound_micro_R1']==0
    permutation=[6,5,4,3,2,1,0]
    assert a==certified_native256_retrieval(e[permutation],[v[i] for i in permutation],[l[i] for i in permutation])


def test_native_finite_nonzero_complete_shape_no_padding_or_unlabeled_query():
    e,v,l=example()
    with pytest.raises(ValueError):certified_native256_retrieval(np.pad(e,((0,0),(0,512))),v,l)
    e[0]=0
    with pytest.raises(ValueError):certified_native256_retrieval(e,v,l)
    e,v,l=example();e[0,1]=np.nan
    with pytest.raises(ValueError):certified_native256_retrieval(e,v,l)


def test_full_physical_native_metrics_geometry_strata_and_input_immutability():
    e,v,l=example();raw=np.pad(e,((0,0),(0,512)));p=np.eye(5,256);r=np.pad(p,((0,0),(0,512)))
    tracks=[SimpleNamespace(video=video,count=lambda i=i:(1 if i<2 else 5)) for i,video in enumerate(v)]
    original=[a.copy() for a in (raw,e,p,r)]
    d=evaluate_prefix(raw,e,tracks,l,{20:0,21:1,22:2,23:3,24:4},r,p,[1]*7)
    assert d['Raw_drift']['full_pairwise_Gram_MSE']==0
    assert d['Raw_drift']['pointwise_Raw_cosine_not_defined_across_768D_and_256D']
    assert d['fresh_A']['singleton_queries_full_gallery']['physical_gallery_tracks']==7
    assert d['fresh_A']['singleton_queries_full_gallery']['declared_query_tracks']==2
    assert d['support']['full_physical_tracks']==7
    for a,b in zip((raw,e,p,r),original):assert np.array_equal(a,b)
    with pytest.raises(ValueError):evaluate_prefix(raw,e,tracks,l,{20:0},r,p,[1]*7)
    with pytest.raises(ValueError):evaluate_prefix(raw,e,tracks,l,{20:0,21:1,22:2,23:3,24:4},r,p,[100]*7)
