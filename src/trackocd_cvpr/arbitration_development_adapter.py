"""External dev bindings to unchanged observer buckets, never model labels."""
from .arbitration_attribution import attribution_stream


def development_targets(labels, fit_bindings):
    targets = {}
    for key,(kind,category) in labels.items():
        if kind == 'Known':
            if category not in fit_bindings:raise ValueError('Existing offline fit Known binding, not dev prototype')
            targets[key] = ('Known',fit_bindings[category])
        elif kind == 'declared_pseudoNovel':
            if type(category) is not int or category < 0 or category in fit_bindings:
                raise ValueError('Disjoint existing Known-derived development pseudo binding')
            # Legacy observer's bucket spelling only. Preserve external binding
            # value, no training target, GT to scorer, or new Novel vocabulary.
            targets[key] = ('policy_pseudo',category)
        elif kind in ('outside_eligible_Known','no_retained_semantic_truth') and category is None:
            targets[key] = ('masked',None)
        else:raise ValueError('No new semantic role/value or invented negative from dev masks')
    if len(targets) != len(labels):raise ValueError('Full physical external target dictionary')
    return targets


def development_stream(videos, labels, known, fit_bindings, cfg, order, guard, *, head,
                       observe=True, rename_and_reverse=False):
    return attribution_stream(videos,development_targets(labels,fit_bindings),known,cfg,order,guard,
        head=head,observe=observe,rename_and_reverse=rename_and_reverse)
