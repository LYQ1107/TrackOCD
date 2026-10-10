"""Conditional paired video resampling, never a causal-memory/GT repair."""
import numpy as np


def paired_video_bootstrap(left,right,samples=500,seed=1027):
    a={r['video_id']:r for r in left};b={r['video_id']:r for r in right}
    if set(a)!=set(b):raise ValueError('Paired bootstrap requires identical video universe')
    keys=sorted(a);rng=np.random.default_rng(seed)
    fields=('known_gt','novel_gt','old_correct','new_correct','reuse_opportunities','correct_ct')
    arrays=[np.asarray([[data[v][f] for f in fields] for v in keys],dtype=np.float64) for data in (a,b)]
    def score(c):
        old=c[2]/c[0] if c[0] else np.nan;new=c[3]/c[1] if c[1] else np.nan
        h=2*old*new/(old+new) if old+new>0 else 0. if np.isfinite(old+new) else np.nan
        ct=c[5]/c[4] if c[4] else np.nan
        return np.asarray([h,ct])
    deltas=[]
    for _ in range(samples):
        ix=rng.integers(len(keys),size=len(keys));deltas.append(score(arrays[1][ix].sum(0))-score(arrays[0][ix].sum(0)))
    deltas=np.asarray(deltas);observed=score(arrays[1].sum(0))-score(arrays[0].sum(0));result={}
    for i,name in enumerate(('h_score','correct_commit_ct')):
        values=deltas[np.isfinite(deltas[:,i]),i]
        result[name]={'right_minus_left':float(observed[i]) if np.isfinite(observed[i]) else None,
            'conditional_video_bootstrap_95pct':np.quantile(values,[.025,.975]).tolist() if len(values) else None,'valid_resamples':len(values)}
    return {'metrics':result,'samples':samples,'seed':seed,'videos':len(keys),
        'conditional_on_frozen_global_mapping_and_predicted_stream_state':True,
        'independent_video_or_category_observations_claimed':False,
        'limitation':'Video/category dependence and sparse pseudo classes; not a replay-level causal confidence interval'}
