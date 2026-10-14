"""Read-only observer of prediction-owned support; no empirical result here.

Shadow scores use ONE actual pre-frame history, not a counterfactual rollout.
Opaque routes/epochs are bookkeeping for membership exclusion only; never
passed to the numeric scorer. GT is external and applied after predictions.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import numpy as np

from .atomic_member_control import AtomicMemberControl
from .causal_observation_router import CurrentObservationRouter, visible_bytes
from .candidate_score_contract import build_candidates, resolve_scores, unit
from .prefix_member_bank import StateEvidence
from .policy_training_data import policy_current_frames
from .raw_known_membership_control import calibrated_raw_scores, parameter_sha256


@dataclass(frozen=True)
class ObservedSupport:
    epoch: int
    frame: int
    token: str
    payload: bytes
    vector_bytes: bytes
    quality: float


class PredictionSupportObserver:
    """Separate append-only output witness, never a bank/scorer command API."""
    def __init__(self, max_members, max_tokens, max_commits, max_payload_bytes=67108864):
        self.bounds = (max_members, max_tokens, max_commits, max_payload_bytes)
        self.epoch = 0; self.members = {}; self.tokens = set(); self.commits = 0

    def next_video(self):
        self.epoch += 1

    def evidence(self, *, exclude_aliases=()):
        excluded = {(self.epoch, route) for route in exclude_aliases}
        by_token = {token:{} for token in self.tokens}; count = Counter()
        for key, support in self.members.items():
            if key in excluded:continue
            by_token[support.token][(support.epoch, support.frame, support.payload)] = support
            count[support.token] += 1
        out = {}
        for token, groups in by_token.items():
            total = np.zeros(768, dtype='<f8'); quality = 0.
            for key in sorted(groups):
                s = groups[key]; total += np.frombuffer(s.vector_bytes,dtype='<f4'); quality += s.quality
            n = len(groups); norm = float(np.linalg.norm(total))
            status = 'EMPTY' if not n else 'ZERO_RESULTANT' if norm <= 1e-12 else 'DEFINED'
            out[token] = StateEvidence(token,n,count[token],total.tobytes(),quality/n if n else 0.,
                float(np.clip(1-norm/n,0,1)) if n else 0.,status)
        return out

    def membership(self, token, aliases):
        own = {(self.epoch, route) for route in aliases}
        keys = [key for key,s in self.members.items() if s.token == token]
        if not keys:raise ValueError('Actual selected Existing token must have prior support')
        return {'contains_own':any(k in own for k in keys),'only_own':all(k in own for k in keys),
                'has_other_same_video':any(k[0] == self.epoch and k not in own for k in keys),
                'has_other_video':any(k[0] != self.epoch for k in keys)}

    def observe(self, batch, result):
        if result.routing_groups != batch.routing_groups or len(result.events) != len(batch.model_views):
            raise ValueError('Actual input/event/alias alignment, no observer-specific routing')
        for view, aliases, event in zip(batch.model_views,batch.routing_groups,result.events):
            if event.token is None:continue  # Preserve actual Known/WAIT retention semantics.
            support = ObservedSupport(self.epoch,batch.frame_index,event.token,visible_bytes(view),
                view.weighted_mean().astype('<f4').tobytes(),float(view.quality.mean()))
            self.tokens.add(event.token)
            for route in aliases:self.members[(self.epoch,route)] = support; self.commits += 1
        payloads = {s.payload for s in self.members.values()}
        if (len(self.members)>self.bounds[0] or len(self.tokens)>self.bounds[1] or self.commits>self.bounds[2]
                or sum(len(p) for p in payloads)>self.bounds[3]):
            raise ValueError('Observer bounds, no dropped support/unknown/history')


def external_label(targets, video, aliases):
    labels = [targets.get((video,route),('masked',None)) for route in aliases]
    if not labels or len(set(labels)) != 1:return None
    kind, binding = labels[0]
    if kind not in ('Known','policy_pseudo'):return None
    if type(binding) is not int or binding < 0:raise ValueError('Existing eligible policy binding only')
    return kind,binding


def attribution_stream(videos, targets, known, cfg, order, guard, *, head, observe=True, rename_and_reverse=False):
    """Full uncapped policy stream, unchanged controller and scalar head.

    Returns public aggregates plus PRIVATE actual event/commit witnesses for
    caller parity tests. No per-object/semantic/feature rows may be exported.
    Exclusion is a same-history diagnostic score only; never executes a write.
    """
    if head.training or any(p.requires_grad for p in head.parameters()):raise ValueError('Frozen head, not an optimizer')
    before = parameter_sha256(head); counts = Counter(); strata = {k:Counter() for k in ('Known','policy_pseudo','masked')}
    control = AtomicMemberControl(known,lambda f,c:calibrated_raw_scores(f,c,model=head),K=8,
        bank_limits={'max_members':cfg['physical_tracks'],'max_tokens':cfg['observations'],
                     'max_commits':cfg['observations'],'max_payload_bytes':67108864})
    # Mirror EXACT constructor normalization before build_candidates applies
    # its own unit(). A near-unit float32 prototype need not be unchanged by
    # repeated normalization; shadow differences must not come from that.
    candidate_known = {key:unit(value).copy() for key,value in known.items()}
    witness = PredictionSupportObserver(cfg['physical_tracks'],cfg['observations'],cfg['observations']) if observe else None
    private = []; sequence = np.random.default_rng(order).permutation(len(videos)).tolist(); routes = {}
    if rename_and_reverse:
        for video,arrays in videos:
            routes[video] = {int(t):'renamed-'+str(len(arrays['track_id'])-i-1) for i,t in enumerate(arrays['track_id'])}
        targets = {(v,routes[v][int(route)]):label for (v,route),label in targets.items()}
    for position, ordinal in enumerate(sequence):
        video, arrays = videos[ordinal]; router = CurrentObservationRouter()
        if position:
            control.start_next_video()
            if witness:witness.next_video()
        for frame, rows in policy_current_frames(arrays,routing_bindings=routes.get(video)):
            if rename_and_reverse:rows = tuple(reversed(rows))
            guard(); batch = router.ingest(frame,rows) if witness else None
            if witness:
                actual_bank,_ = control.state_snapshot()
                if witness.evidence() != actual_bank.state_evidence():raise ValueError('Exact actual pre-frame numeric support witness')
                del actual_bank
            result = control.ingest(frame,rows); private.append((video,result))
            counts['metadata_frames'] += 1; counts['empty_frames'] += int(not rows); counts['physical_observations'] += len(rows)
            for i,(aliases,event) in enumerate(zip(result.routing_groups,result.events)):
                counts['canonical_groups'] += 1; counts[event.action] += len(aliases)
                # External label AFTER predictions. Never supplied to callback,
                # observer support construction, or counterfactual candidates.
                label = external_label(targets,video,aliases); bucket = strata[label[0] if label else 'masked']
                bucket['canonical_groups'] += 1; bucket['physical_observations'] += len(aliases); bucket[event.action] += len(aliases)
                if not witness:continue
                current = build_candidates(batch.model_views[i],candidate_known,witness.evidence(),K=8)
                if (current.feature_bytes != event.candidate_feature_bytes or current.context_bytes != event.query_context_bytes
                        or current.bindings != event.candidate_bindings_for_external_loss):
                    raise ValueError('Exact actual pre-frame candidate numeric/binding witness, not shadow normalization drift')
                cf,cc = current.numeric_inputs()
                if calibrated_raw_scores(cf,cc,model=head).astype('<f4').tobytes() != event.score_bytes:
                    raise ValueError('Exact actual unchanged numeric score witness')
                f = np.frombuffer(event.candidate_feature_bytes,dtype='<f4').reshape(-1,11)
                bk = float(f[f[:,1] == 1,0].max()); positive = float(head(bk)) >= 0
                bucket['named_gate_positive_groups'] += int(positive)
                if event.action != 'EXISTING':continue
                membership = witness.membership(event.token,aliases)
                for k,v in membership.items():bucket['Existing_'+k+'_groups'] += int(v)
                partition = 'only_own' if membership['only_own'] else 'own_and_other' if membership['contains_own'] else 'other_only'
                bucket['Existing_provenance_partition_'+partition+'_groups'] += 1
                bucket['positive_named_gate_bypassed_groups'] += int(positive)
                snapshot = build_candidates(batch.model_views[i],candidate_known,witness.evidence(exclude_aliases=aliases),K=8)
                nf,nc = snapshot.numeric_inputs(); action,binding,_ = resolve_scores(snapshot,calibrated_raw_scores(nf,nc,model=head))
                bucket['same_history_remove_own_changes_action_groups'] += int(action != event.action)
                bucket['same_history_remove_own_KNOWN_groups'] += int(action == 'KNOWN')
                bucket['same_history_remove_own_correct_Known_groups'] += int(label is not None and label[0] == 'Known' and action == 'KNOWN' and binding == label[1])
            if witness:
                witness.observe(batch,result); actual_bank,_ = control.state_snapshot()
                if witness.evidence() != actual_bank.state_evidence() or witness.commits != len(actual_bank.commit_history()):
                    raise ValueError('Exact actual post-frame numeric support/history count witness')
                del actual_bank
    if (counts['physical_observations'],counts['metadata_frames'],counts['empty_frames']) != (cfg['observations'],cfg['metadata_frames'],cfg['empty_frames']):
        raise ValueError('Full policy clock/unknown/empty observations, not successful subset')
    if parameter_sha256(head) != before:raise ValueError('Read-only head changed')
    bank,allocated = control.state_snapshot()
    return {'order_seed':order,'video_ordinal_schedule_sha256':hashlib.sha256(np.asarray(sequence,dtype='<i8').tobytes()).hexdigest(),
        'full_clock':dict(counts),'external_strata':{k:dict(v) for k,v in strata.items()},
        'allocated_tokens':allocated,'actual_commits':len(bank.commit_history()),
        'exact_numeric_support_witness_checked':observe,'model_state_unchanged':True,
        'same_history_one_decision_shadow_not_counterfactual_rollout_gain':True,
        'GT_or_shadow_decisions_used_for_any_policy_or_memory_write':False}, (private,bank.commit_history(),bank.state_evidence())
