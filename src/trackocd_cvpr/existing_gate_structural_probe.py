"""Deterministic SYNTHETIC source diagnostic, NOT actual dev attribution.

No GT/data/checkpoint access, optimizer, sweep, control change or repair. The
registered four-control real window remains byte-for-byte unchanged.
"""
import numpy as np
from .atomic_member_control import AtomicMemberControl
from .causal_observation_router import CurrentObservation
from .raw_known_membership_control import NamedKnownLogit, calibrated_raw_scores


def observation(cosine, second):
    x = np.zeros(768, np.float32); x[:2] = [cosine, second]
    return CurrentObservation('opaque-route', x, np.array([.1, .1, .5, .5], np.float32), 1.)


def known():
    x = np.zeros(768, np.float32); x[0] = 1.
    return {0:x}


def fixed_head():
    return NamedKnownLogit().eval().requires_grad_(False)


def known_gate_bypassed_by_own_prior_support():
    """A .6 initial prefix births an anonymous state; later evidence tends
    toward the Known prototype, but own-prefix Existing arbitration wins.
    This is a counterexample, not a measured frequency or a proposed repair.
    """
    head = fixed_head()
    score = lambda f,c:calibrated_raw_scores(f,c,model=head)
    control = AtomicMemberControl(known(), score, K=8)
    frames = []
    for frame in range(1, 10):
        row = observation(.6, .8) if frame == 1 else observation(1., 0.)
        result = control.ingest(frame, (row,)); event = result.events[0]
        f = np.frombuffer(event.candidate_feature_bytes, dtype='<f4').reshape(-1, 11)
        bk = float(f[f[:,1] == 1,0].max()); existing = f[:,1] == 0
        be = float(f[existing,0].max()) if existing.any() else None
        frames.append({'frame':frame, 'max_Known_cosine':bk, 'max_Existing_cosine':be,
            'frozen_named_membership_gate_positive':bool(float(head(bk)) >= 0), 'actual_action':event.action})
    bank, count = control.state_snapshot()
    return {'status':'SYNTHETIC_SOURCE_COUNTEREXAMPLE_NOT_EMPIRICAL_CAUSE_OR_SCIENCE_WIN',
        'frames':frames, 'anonymous_tokens':count, 'actual_commits':len(bank.commit_history()),
        'sole_member_support_groups':[s.support_groups for s in bank.state_evidence().values()],
        'GT_labels_weights_data_or_dev_predictions_read':False, 'registered_real_experiment_rule_or_history_changed':False,
        'actual_dev_attribution_or_Goal_accepted':False}


def known_output_retains_old_anonymous_support():
    """Explicitly forced diagnostic Known output, NOT a new score policy.

    Shows the existing transaction's retention semantics without deleting or
    changing any historical records. No GT or routing ID to numeric scorer.
    """
    head = fixed_head(); calls = 0
    def score(f, c):
        nonlocal calls
        calls += 1
        if calls == 1: return calibrated_raw_scores(f, c, model=head)
        scores = np.full(len(f)+2, -2., dtype=np.float32)
        take = f[:,1] == 1; bk = f[take,0].max()
        scores[np.flatnonzero(take & (f[:,0] == bk))] = 1.
        return scores
    control = AtomicMemberControl(known(), score, K=8)
    first = control.ingest(1, (observation(.6, .8),)); before, _ = control.state_snapshot()
    second = control.ingest(2, (observation(1., 0.),)); after, _ = control.state_snapshot()
    return {'status':'SYNTHETIC_FORCED_OUTPUT_TRANSACTION_DIAGNOSTIC_NOT_NEW_POLICY',
        'first_actual_action':first.events[0].action, 'diagnostic_forced_second_action':second.events[0].action,
        'prior_commits_unchanged_and_retained':before.commit_history() == after.commit_history(),
        'prior_anonymous_numeric_support_retained':before.state_evidence() == after.state_evidence(),
        'GT_history_repair_or_actual_dev_attribution':False}


def already_Known_without_anonymous_birth():
    head = fixed_head()
    control = AtomicMemberControl(known(), lambda f,c:calibrated_raw_scores(f,c,model=head), K=8)
    actions = [control.ingest(frame, (observation(1., 0.),)).events[0].action for frame in range(1, 4)]
    bank, count = control.state_snapshot()
    return {'actions':actions, 'anonymous_tokens':count, 'anonymous_commits':len(bank.commit_history()),
            'self_absorption_claimed_inevitable_for_all_sequences':False}
