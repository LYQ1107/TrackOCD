import numpy as np
from scripts.trackocd_core.analyze_train_representation import role_retrieval


def test_role_retrieval_does_not_hide_known_negatives_or_unsupported_queries():
    routes=[{'key':str(i),'video_id':i} for i in range(5)]
    labels=[{'key':str(i),'category_id':c,'simulation_role':r} for i,(c,r) in enumerate(((1,'known'),(1,'known'),(2,'pseudo_novel'),(2,'pseudo_novel'),(3,'pseudo_novel')))]
    # Pseudo class2's two queries are closer to Known negatives,not each other.
    values=np.asarray([[1,0],[0,1],[1,.01],[.01,1],[-1,-1]],np.float32);values/=np.linalg.norm(values,axis=1,keepdims=True)
    result=role_retrieval(values,routes,labels)
    assert result['pseudo_novel']['query_tracks']==3
    assert result['pseudo_novel']['queries_with_cross_video_positive']==2
    assert result['pseudo_novel']['unsupported_queries']==1
    assert result['pseudo_novel']['recall_at_1_category_macro']==0
    assert result['pseudo_novel']['recall_at_5_category_macro']==1
