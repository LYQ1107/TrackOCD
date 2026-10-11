"""Train evaluator-only future-input shadow, never a deployable wait policy."""
from collections import Counter
import hashlib
import json
from src.trackocd_core.persistent_policy import CandidateMemory, SupervisionHistory, ACTIONS
from src.trackocd_core.train_first_replay import causal_rows
from src.trackocd_core.evaluation import DecisionEvent, TrackKey, seal_decisions
from src.trackocd_cvpr.checkpoint_diagnostics import visual_step


def visual_history_digest(memory):
    h = hashlib.sha256()
    h.update(str(memory.next_id).encode())
    for key, vector in sorted(memory.known.items()):
        h.update(str(key).encode()); h.update(vector.tobytes())
    for token, member in sorted(memory.anonymous.items()):
        h.update(token.encode()); h.update(member.vector_sum.tobytes())
        h.update(json.dumps([member.count, member.quality_sum], allow_nan=False).encode())
    return h.hexdigest()


def fixed_history_replay(routes, order, prefix, proto, bank, future_bank, model, labels, constants, wait_bias):
    memory, observer = CandidateMemory(proto), SupervisionHistory()
    events, rows, summary = [], [], Counter()
    for sequence, route in enumerate(causal_rows(routes, order, prefix)):
        item = bank[route['key']]
        # Current and shadow queries are pure reads of the SAME actual prior visual history.
        before = visual_history_digest(memory)
        action, known, token, _, logits = visual_step(model, memory, item, wait_bias)
        later = future_bank[route['key']]
        has_later = later['actual_prefix'] > item['actual_prefix']
        shadow = visual_step(model, memory, later, wait_bias) if has_later else None
        if before != visual_history_digest(memory):
            raise AssertionError('A current/shadow query mutated actual visual history')
        # GT belongs to the external observer only, AFTER both visual decisions.
        label = labels[route['key']]
        category, role = label['category_id'], label['simulation_role']
        if role not in ('known', 'pseudo_novel'):
            raise ValueError('Registered Train diagnostic roles only')
        costs, _, _ = observer.costs(category, role == 'known', known, token, item['actual_prefix'], constants)
        wrong_commit = action != 3 and costs[action] > 0
        row = {'sequence': sequence, 'physical_key': route['key'], 'actual_prefix': item['actual_prefix'],
            'actual_action': ACTIONS[action], 'actual_logits': logits.tolist(), 'actual_nominal_cost': float(costs[action]),
            'prior_visual_history_sha256': before, 'later_input_available': has_later,
            'role_evaluator_only': role, 'future_prefix_evaluator_only': None,
            'future_shadow_action_evaluator_only': None, 'future_shadow_nominal_cost_evaluator_only': None}
        summary[f'{role}_queries'] += 1
        summary[f'{role}_actual_wrong_commits'] += int(wrong_commit)
        summary[f'{role}_later_input_available'] += int(has_later)
        if has_later:
            shadow_action, shadow_known, shadow_token, _, _ = shadow
            shadow_costs, _, _ = observer.costs(category, role == 'known', shadow_known, shadow_token,
                                               later['actual_prefix'], constants)
            row.update({'future_prefix_evaluator_only': later['actual_prefix'],
                'future_shadow_action_evaluator_only': ACTIONS[shadow_action],
                'future_shadow_nominal_cost_evaluator_only': float(shadow_costs[shadow_action])})
            summary[f'{role}_wrong_commits_with_later_input'] += int(wrong_commit)
            summary[f'{role}_wrong_commit_later_zero_nominal_cost'] += int(wrong_commit and shadow_action != 3 and shadow_costs[shadow_action] == 0)
            summary[f'{role}_wrong_commit_later_WAIT'] += int(wrong_commit and shadow_action == 3)
            summary[f'{role}_wrong_commit_later_still_wrong_commit'] += int(wrong_commit and shadow_action != 3 and shadow_costs[shadow_action] > 0)
            summary[f'{role}_correct_commit_later_wrong_commit'] += int(action != 3 and costs[action] == 0 and shadow_action != 3 and shadow_costs[shadow_action] > 0)
        # ONLY the original current-input decision writes visual and external histories.
        prediction = memory.apply(action, known, token, item['embedding'], item['quality'])
        observer.update(prediction, category, sequence)
        events.append(DecisionEvent(sequence, TrackKey(route['video_id'], str(route['physical_track_id'])),
            item['actual_prefix'], prediction['kind'], known_category_id=prediction.get('category_id'), token=prediction.get('token')))
        rows.append(row)
    for role in ('known', 'pseudo_novel'):
        if summary[f'{role}_wrong_commits_with_later_input'] != sum(summary[f'{role}_wrong_commit_later_{suffix}']
                for suffix in ('zero_nominal_cost', 'WAIT', 'still_wrong_commit')):
            raise AssertionError('Future shadow outcomes must partition actual wrong commits with later input')
    return seal_decisions(events, video_order=order, prefix_cap=prefix, known_ids=tuple(proto)), rows, dict(summary)
