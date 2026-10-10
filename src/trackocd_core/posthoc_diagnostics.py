"""Evaluator-only error flags; no mapping/GT is returned to a live policy."""
from __future__ import annotations
import numpy as np
from src.trackocd_core.evaluation import evaluate_standard,evaluate_persistent
from src.trackocd_core.evaluation.persistent import fixed_cross_video_targets,_History


def evaluate_with_flags(join):
    standard=evaluate_standard(join);mapping=standard.pop('global_anonymous_hungarian_mapping_evaluator_only')
    persistent=evaluate_persistent(join)
    targets=sorted((t for t in join.targets if t.role!='distractor'),key=lambda t:(t.key.video_id,t.key.local_track_id))
    by_key={t.key:t for t in join.targets};pred_to_gt=dict(join.matches);gt_to_pred={g:p for p,g in join.matches if g is not None}
    commits={c.event.physical_key:c.event for c in join.replay.commits}
    eligible={t.key for t in fixed_cross_video_targets(join.targets,join.replay.video_order)}
    ct_keys=set();histories={};pollution_writes=0;unknown_writes=0
    for c in join.replay.commits:
        e=c.event;t=by_key.get(pred_to_gt.get(e.physical_key));history=None
        if e.kind=='NEW':history=histories[e.token]=_History()
        elif e.kind=='EXISTING':history=histories[e.token]
        if t is not None and t.key in eligible and e.kind=='EXISTING':
            if (not history.mixed and not history.unknown and history.first_category==t.category_id
                and (history.multiple_videos or history.first_video!=t.key.video_id)):ct_keys.add(t.key)
        if history is not None:
            if history.members and t is not None and (history.mixed or history.first_category is not None and history.first_category!=t.category_id):pollution_writes+=1
            unknown_writes+=t is None;history.add(t,e.physical_key.video_id)
    if len(ct_keys)!=persistent['commit_ct_correct']:raise AssertionError('Error flags must exactly match audited CT,never HungarianCT')
    per_video={v:{'video_id':v,'known_gt':0,'novel_gt':0,'old_correct':0,'new_correct':0,'reuse_opportunities':0,'correct_ct':0} for v in join.replay.video_order}
    hits=[];ct_hits=[];reuse=[];matched=[];novel=[];wrong_known=0;matched_known=0;matched_novel=0
    for t in targets:
        p=gt_to_pred.get(t.key);e=commits.get(p)
        hit=bool(e is not None and (e.kind=='KNOWN' and e.known_category_id==t.category_id if t.role=='known'
                   else e.kind in {'NEW','EXISTING'} and mapping.get(e.token)==t.category_id))
        row=per_video[t.key.video_id];row['known_gt' if t.role=='known' else 'novel_gt']+=1
        row['old_correct' if t.role=='known' else 'new_correct']+=hit;row['reuse_opportunities']+=t.key in eligible;row['correct_ct']+=t.key in ct_keys
        matched_known+=t.role=='known' and p is not None;matched_novel+=t.role=='novel' and p is not None
        wrong_known+=t.role=='novel' and e is not None and e.kind=='KNOWN'
        hits.append(hit);ct_hits.append(t.key in ct_keys);reuse.append(t.key in eligible);matched.append(p is not None);novel.append(t.role=='novel')
    if sum(r['old_correct'] for r in per_video.values())!=standard['old_correct'] or sum(r['new_correct'] for r in per_video.values())!=standard['new_correct']:raise AssertionError('Exact single-global-mapping diagnostic counts')
    old=standard['old_correct']/matched_known if matched_known else None;new=standard['new_correct']/matched_novel if matched_novel else None
    extra={'all_novel_wrong_known_count':int(wrong_known),'all_novel_wrong_known_denominator':standard['new_denominator'],
        'all_novel_wrong_known_rate':wrong_known/standard['new_denominator'] if standard['new_denominator'] else None,
        'memory_contamination_write_events':int(pollution_writes),'contaminated_states':sum(h.mixed for h in histories.values()),
        'unmatched_anonymous_write_events':int(unknown_writes),'states_with_unknown_members':sum(h.unknown for h in histories.values()),
        'unknown_members_not_proven_pollution_but_cannot_certify_purity':True,
        'per_video_conditional_fixed_mapping_counts':list(per_video.values()),
        'matched_only_diagnostic':{'known_denominator':matched_known,'novel_denominator':matched_novel,'old_acc':old,'new_acc':new,
            'h_score':2*old*new/(old+new) if old is not None and new is not None and old+new else 0. if old is not None and new is not None else None,
            'same_global_mapping_no_second_Hungarian':True,'not_primary_full_GT_result':True}}
    flags={'video_id':np.asarray([t.key.video_id for t in targets],np.int32),
        'local_id':np.asarray([t.key.local_track_id for t in targets]),'standard_correct':np.asarray(hits,bool),
        'CT_correct':np.asarray(ct_hits,bool),'reuse_opportunity':np.asarray(reuse,bool),'physically_matched':np.asarray(matched,bool),'novel':np.asarray(novel,bool)}
    return {'standard':standard,'persistent':persistent,'errors':extra},flags


def paired_corrections(left,right):
    for name in ('video_id','local_id','reuse_opportunity','novel'):
        if not np.array_equal(left[name],right[name]):raise ValueError('Paired correction universe differs')
    result={}
    for metric in ('standard_correct','CT_correct'):
        a,b=left[metric],right[metric];mask=left['reuse_opportunity'] if metric=='CT_correct' else np.ones(len(a),bool)
        for role,role_mask in (('all',mask),('known',mask&~left['novel']),('novel',mask&left['novel'])):
            result[f'{metric}_{role}']={'denominator':int(role_mask.sum()),'errors_corrected':int((~a&b&role_mask).sum()),
                'new_errors_introduced':int((a&~b&role_mask).sum()),'both_wrong':int((~a&~b&role_mask).sum()),'both_correct':int((a&b&role_mask).sum()),
                'net_correct_change':int(((~a&b&role_mask).sum())-((a&~b&role_mask).sum()))}
    return result
