"""Native256D certified retrieval, not768 padding or cross-space Raw cosine.

Same external labels/full-gallery/exclude-whole-video/exact-tie semantics as
the frozen768D R diagnostic. Caller must gate complete15fit cohort and legal
full dev loader BEFORE actual inputs. No model, checkpoint or asset reads.
"""
from collections import defaultdict
import numpy as np
from .R_development_metrics import _certified_topk,certified_cross_video_retrieval
from .R_development_evaluation import known_recognition,effective_rank
from .fresh_A_baseline import cross_space_Gram_drift


def normalized(vectors,n,d):
    x=np.array(vectors,dtype=np.float64,copy=True)
    if x.shape!=(n,d) or not n or not np.isfinite(x).all():raise ValueError('Complete aligned finite native embeddings')
    norm=np.linalg.norm(x,axis=1)
    if np.any(norm<=1e-12):raise ValueError('No undefined embedding or dropped physical row')
    return x/norm[:,None]


def certified_native256_retrieval(embeddings,videos,external_labels,query_kind='declared_pseudoNovel'):
    videos=tuple(videos);labels=tuple(external_labels);n=len(videos)
    if len(labels)!=n:raise ValueError('Complete physical gallery/external labels alignment')
    scores=normalized(embeddings,n,256)@normalized(embeddings,n,256).T
    values=defaultdict(list);classes=defaultdict(lambda:defaultdict(list));no_positive=unknown_top=queries=0
    for i,(kind,category) in enumerate(labels):
        if kind!=query_kind:continue
        if category is None:raise ValueError('Lawful declared query truth required, not invented unknown target')
        gallery=np.array([j for j in range(n) if videos[j]!=videos[i]],dtype=np.int64)
        similarity=scores[i,gallery]
        correct=np.array([labels[j][1] is not None and labels[j][1]==category for j in gallery],bool)
        if not correct.any():no_positive+=1
        if len(gallery):
            top=gallery[similarity==similarity.max()]
            unknown_top+=int(any(labels[j][1] is None for j in top))
        for k in (1,5):
            value=_certified_topk(similarity,correct,k);values[k].append(value);classes[category][k].append(value)
        queries+=1
    if not queries:raise ValueError('No frozen declared query population')
    return {'physical_gallery_tracks':n,'declared_query_tracks':queries,'declared_query_classes':len(classes),
        'queries_without_verified_cross_video_positive_retained':no_positive,
        'queries_with_semantic_unknown_in_top_tie':unknown_top,
        'certified_lower_bound_micro_R1':float(np.mean(values[1])),
        'certified_lower_bound_micro_R5':float(np.mean(values[5])),
        'certified_lower_bound_macro_R1':float(np.mean([np.mean(v[1]) for v in classes.values()])),
        'certified_lower_bound_macro_R5':float(np.mean([np.mean(v[5]) for v in classes.values()])),
        'samevideo_excluded':True,'unknown_gallery_declared_background_or_labeled_Novel':False,
        'exact_tie_uniform_expectation_no_ID_ordering':True,
        'not_complete_semantic_GT_purity_Hscore_or_final_acceptance':True}


def evaluate_prefix(raw,adapted,tracks,labels,bindings,raw_prototypes,adapted_prototypes,actual):
    n=len(tracks);videos=[t.video for t in tracks]
    if len(labels)!=n or len(actual)!=n or any(type(k) is not int or not 1<=k<=t.count() for t,k in zip(tracks,actual)):
        raise ValueError('Full identical physical observed prefix footprint, no padding/future')
    if len(bindings)!=5 or set(bindings.values())!=set(range(5)):
        raise ValueError('Exactly offline fit5 Known prototypes, no dev prototype')
    normalized(raw,n,768);normalized(adapted,n,256)
    normalized(raw_prototypes,5,768);normalized(adapted_prototypes,5,256)
    result={}
    for name,embedding,prototypes,retrieval in (
            ('Raw',raw,raw_prototypes,certified_cross_video_retrieval),
            ('fresh_A',adapted,adapted_prototypes,certified_native256_retrieval)):
        result[name]={'retrieval':retrieval(embedding,videos,labels),
            'Known':known_recognition(np.array(embedding,copy=True),labels,bindings,np.array(prototypes,copy=True)),
            'effective_rank_full_physical':effective_rank(embedding)}
        for stratum,predicate in (('singleton',lambda t:t.count()==1),('short_le4',lambda t:t.count()<=4)):
            scoped=[(kind if predicate(t) else 'gallery_only',category) for t,(kind,category) in zip(tracks,labels)]
            result[name][stratum+'_queries_full_gallery']=(retrieval(embedding,videos,scoped)
                if any(kind=='declared_pseudoNovel' for kind,_ in scoped)
                else {'declared_query_tracks':0,'status':'NO_QUERY_STRATUM_NOT_FILTERED_MAIN'})
    result['Raw_drift']=cross_space_Gram_drift(np.asarray(raw),np.asarray(adapted))
    result['support']={'full_physical_tracks':n,'visible_observations_forwarded':sum(actual),
        'singleton_tracks_kept':sum(t.count()==1 for t in tracks),'short_le4_tracks_kept':sum(t.count()<=4 for t in tracks),
        'full_physical_gallery_short_and_unlabeled_not_dropped':True}
    result['feature_spaces']={'Raw_dimensions':768,'fresh_A_dimensions':256,
        'zero_padding_or_undefined_cross_space_pointwise_cosine_used':False}
    return result
