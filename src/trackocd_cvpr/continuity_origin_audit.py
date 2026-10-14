"""External origin-aware GT-control audit; no feedback to controller or loss.

The unchanged conservative history audit remains a separate identity diagnostic.
Primary scored-reuse counts mask routing-only continuation, while retaining ALL
physical/unknown denominators. No full semantic GT CT/H-score or Pred claim.
"""
from collections import Counter
from dataclasses import replace
import numpy as np
from .causal_category_audit import CausalCategoryAudit
from .continuity_reuse_control import ContinuityReceipt


class ContinuityOriginAudit:
    def __init__(self, labels, known_bindings, *, prefixes=(1,2,4,8,16), max_tokens=1268):
        self.history = CausalCategoryAudit(labels,known_bindings,prefixes=prefixes,max_tokens=max_tokens)
        self._origins, self._last = Counter(), {}
        self._snapshots = {n:{} for n in prefixes}
        self._first_commit, self._first_scored, self._first_certified_scored = {}, {}, {}

    def audit_frame(self, video, result, receipts):
        if len(receipts)!=len(result.events) or any(type(r) is not ContinuityReceipt for r in receipts):
            raise ValueError('Every actual prediction has its origin receipt')
        for e,r in zip(result.events,receipts):
            if r.category_match_or_independent_support_certified:
                raise ValueError('Routing receipt is never a semantic certificate')
            if r.decision_origin=='ROUTE_CONTINUATION_NOT_CATEGORY_MATCH':
                if (e.action!='EXISTING' or e.category_id is not None or r.scored_action!='NEW' or r.scored_binding is not None
                        or len(r.previous_route_tokens)!=1 or e.token!=r.previous_route_tokens[0]):
                    raise ValueError('Explicit unambiguous scoredNEW routing fallback only')
            elif r.decision_origin=='SCORED_CANDIDATE':
                expected=e.category_id if e.action=='KNOWN' else e.token if e.action=='EXISTING' else None
                if r.scored_action!=e.action or r.scored_binding!=expected:
                    raise ValueError('Actual scored binding/action, not relabeled continuation')
            elif r.decision_origin=='CONFLICTING_ROUTE_CONTINUITY':
                if (e.action!='WAIT' or e.token is not None or e.category_id is not None or r.scored_action!='NEW'
                        or r.scored_binding is not None or len(r.previous_route_tokens)<2):
                    raise ValueError('Conflicting route fallback cannot certify a category')
            else:raise ValueError('Finite origin enum only')
        self.history.audit_frame(video,result)  # AFTER predictions; no outputs back to policy.
        for aliases,e,r in zip(result.routing_groups,result.events,receipts):
            for route in aliases:
                key=(video,route);n=self.history._seen[key];decision=self.history._last[key]
                scored=e.action=='EXISTING' and r.decision_origin=='SCORED_CANDIDATE'
                if not scored:decision=replace(decision,certified_other_physical_reuse=False,certified_cross_video_reuse=False)
                row=(decision,r.decision_origin,scored)
                self._last[key]=row
                if n in self._snapshots:self._snapshots[n][key]=row
                self._origins[r.decision_origin]+=1
                if e.action!='WAIT':self._first_commit.setdefault(key,n)
                if scored:self._first_scored.setdefault(key,n)
                if decision.certified_cross_video_reuse:self._first_certified_scored.setdefault(key,n)

    def _report(self, rows):
        result=self.history._report_rows({k:v[0] for k,v in rows.items()})
        pseudo=[v for k,v in rows.items() if self.history._labels[k][0]=='declared_pseudoNovel']
        result['scored_Existing_claim_track_coverage_over_all_declared_pseudo']=sum(v[2] for v in pseudo)/len(pseudo)
        result['routing_only_continuation_declared_pseudo_tracks']=sum(v[1]=='ROUTE_CONTINUATION_NOT_CATEGORY_MATCH' for v in pseudo)
        result['routing_only_continuation_all_physical_tracks']=sum(v[1]=='ROUTE_CONTINUATION_NOT_CATEGORY_MATCH' for v in rows.values())
        result['certified_reuse_excludes_routing_continuation']=True
        return result

    def finish(self):
        legacy=self.history.finish()
        if set(self._last)!=set(self.history._labels):raise ValueError('All physical origin denominators retained')
        snapshots={str(n):self._report({k:self._snapshots[n].get(k,self._last[k]) for k in self._last}) for n in self._snapshots}
        snapshots['all_available']=self._report(self._last)
        def latency(events,universe):
            values=[events[k] for k in universe if k in events]
            return {'units':'observed_updates_1_based_not_dense_frame_or_wall_time',
                'full_track_denominator':len(universe),'events':len(values),'unresolved_tracks_retained':len(universe)-len(values),
                'mean_conditional_on_event':float(np.mean(values)) if values else None,
                'p50_conditional_on_event':float(np.percentile(values,50)) if values else None,
                'p95_conditional_on_event':float(np.percentile(values,95)) if values else None}
        pseudo=[k for k,v in self.history._labels.items() if v[0]=='declared_pseudoNovel']
        return {'primary_scored_reuse_prefix_snapshots':snapshots,'physical_origin_observations':dict(self._origins),
            'legacy_identity_dirty_history_includes_continuation_not_primary_scored_match':legacy,
            'first_commit_latency':latency(self._first_commit,self._last),
            'first_scored_Existing_claim_latency_declared_pseudo':latency(self._first_scored,pseudo),
            'first_certified_scored_cross_video_reuse_latency_declared_pseudo':latency(self._first_certified_scored,pseudo),
            'GT_or_origin_used_to_rewrite_memory_history':False,'full_semantic_CT_Hscore_Pred_or_Goal_accepted':False}
