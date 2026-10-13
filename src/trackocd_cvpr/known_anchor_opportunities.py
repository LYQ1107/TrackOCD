"""Static lawfully labeled policy-frame opportunities, no inference/optimizer.

Known Top8 proposals include all offline fit5. Known-query loss availability
therefore depends on the current Raw alias group/static Known bindings, not
prediction-owned anonymous history. Caller gates approved caches/targets first.
Public output contains counts/ordinal summaries/calendar hashes only, no rows.
"""
import hashlib
import numpy as np
from .causal_observation_router import CurrentObservationRouter
from .candidate_score_contract import build_candidates
from .candidate_training_loss import lawful_label, prediction_owned_loss_plan
from .policy_training_data import policy_current_frames


def audit_opportunities(videos, projection, known, guard, *, seed=1027, budget=50):
    if type(seed) is not int or seed!=1027 or type(budget) is not int or budget<1 or not 1<=len(known)<=8:
        raise ValueError('Fixed original pilot seed/budget/static full Known family')
    targets={(v,str(t)):label for (v,t),label in projection.targets.items()}
    order=np.random.default_rng(seed).permutation(len(videos)).tolist()
    totals={k:0 for k in ('metadata_frames','empty_frames','physical_observations',
        'Known_eligible_frames','pseudo_eligible_frames','cooccurring_Known_pseudo_frames',
        'Known_groups','pseudo_groups','masked_groups','pseudo_groups_on_all_Known_anchor_frames',
        'selected_update_frames','selected_Known_groups','selected_pseudo_groups')}
    calendar=hashlib.sha256(); records=[]
    for position,ordinal in enumerate(order):
        video,arrays=videos[ordinal]; router=CurrentObservationRouter()
        counts={k:0 for k in totals}
        for frame,rows in policy_current_frames(arrays):
            guard(); batch=router.ingest(frame,rows); nk=npseudo=masked=0
            for view,aliases in zip(batch.model_views,batch.routing_groups):
                keys=[(video,r) for r in aliases]
                labels=[lawful_label(targets.get(k,('no_retained_semantic_truth',None))) for k in keys]
                if any(v is None for v in labels) or len(set(labels))!=1:
                    masked+=1; continue
                candidates=build_candidates(view,known,{},K=8)
                if labels[0][0]=='Known':
                    plan=prediction_owned_loss_plan(candidates.bindings,keys,targets,{})
                    if not any(plan.positive): raise ValueError('Static full Known proposal cannot silently lose target')
                    nk+=1
                else:
                    if not any(any(b.kind=='KNOWN' for b in g) for g in candidates.bindings):
                        raise ValueError('Offline Known set required for rejection opportunity')
                    npseudo+=1
            selected=nk>0 and totals['selected_update_frames']+counts['selected_update_frames']<budget
            values={'metadata_frames':1,'empty_frames':int(not rows),'physical_observations':len(rows),
                'Known_eligible_frames':int(nk>0),'pseudo_eligible_frames':int(npseudo>0),
                'cooccurring_Known_pseudo_frames':int(nk>0 and npseudo>0),
                'Known_groups':nk,'pseudo_groups':npseudo,'masked_groups':masked,
                'pseudo_groups_on_all_Known_anchor_frames':npseudo if nk else 0,
                'selected_update_frames':int(selected),'selected_Known_groups':nk if selected else 0,
                'selected_pseudo_groups':npseudo if selected else 0}
            for key,value in values.items(): counts[key]+=value
            if selected: calendar.update(np.asarray([0,position,frame],dtype='<i8').tobytes())
        for key,value in counts.items(): totals[key]+=value
        records.append({'permutation_position':position,'source_video_ordinal':ordinal,'counts':counts})
    return {'scope_counts':totals,'per_video_ordinal_counts':records,
        'original_first50_known_calendar_sha256':calendar.hexdigest(),
        'full_clock_retained':True,'training_or_prediction_or_extra_semantic_values_read':False,
        'counts_are_opportunities_not_independent_samples_or_gradients':True,
        'calendar_or_model_or_checkpoint_selected':False}
