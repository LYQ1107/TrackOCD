from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
import numpy as np
import pytest
from src.trackocd_cvpr.C_historical_loss_plan_audit import audit_recorded_C_trial
from src.trackocd_cvpr.candidate_scientific_training import scientific_candidate_fit
from tests.trackocd_cvpr.test_policy_training_data import arrays


def synthetic_trial(variant='Top8_conditioned'):
    videos = [(1, arrays())]
    projection = SimpleNamespace(targets={(1, 12): ('Known', 0), (1, 99): ('policy_pseudo', 0)})
    known = {0: np.eye(1, 768, dtype=np.float32)[0]}
    window = {'variants': {variant: {'K_per_family': 1 if variant == 'Top1_conditioned' else 8}},
              'training': {'learning_rate': .001, 'weight_decay': .0001, 'gradient_clip': 1.,
                           'episodes': 2, 'updates_per_episode': 1, 'optimizer_steps': 2, 'checkpoint_steps': [1, 2]}}
    _, _, summary, episodes = scientific_candidate_fit(videos, projection, known, window, variant, 1027,
        lambda: None, lambda r, c: None, lambda *args: None)
    # Exact original writer schema, using genuinely fitted synthetic actions;
    # this fixture is NOT a real dataset, full C cohort or new experiment.
    saved = []
    for episode, (order, sealed, history) in enumerate(episodes):
        saved.append({'episode': episode, 'video_ordinal_order': order,
            'anonymous_commit_labels': [asdict(row) for row in history],
            'actual_predictions': [{'video_id': video, 'frame_index': row.frame_index,
                'routing_groups': row.routing_groups,
                'events': [{'action': event.action, 'category_id': event.category_id, 'token': event.token,
                            'ambiguity_reason': event.ambiguity_reason} for event in row.events]} for video, row in sealed]})
    return videos, {(v, str(t)): label for (v, t), label in projection.targets.items()}, known, window, json.loads(json.dumps(saved)), summary


@pytest.mark.parametrize('variant', ['Top1_conditioned', 'Top8_conditioned', 'Top8_pool_context'])
def test_full_synthetic_history_calendar_tail_and_losses_reconstruct_without_model(variant):
    videos, targets, known, window, saved, summary = synthetic_trial(variant)
    out = audit_recorded_C_trial(videos, targets, known, window, variant, 1027, saved, summary, lambda: None)
    assert out['all_original_episode_summaries_update_calendars_and_retained_histories_match']
    assert out['historical_optimizer_frames_matched'] == 2
    assert out['update_positive_plan_groups_by_family_and_kind']['Known:KNOWN'] == 2
    assert out['update_positive_plan_groups_by_family_and_kind']['policy_pseudo:NEW'] == 2
    assert not out['historical_logits_gradients_or_rowwise_numeric_trace_recovered']
    assert not out['actual_model_inference_or_GT_memory_repair']
    assert out['new_optimizer_steps'] == 0
    assert all(row['metadata_frames_matched'] == 4 for row in out['episodes'])


@pytest.mark.parametrize('fault', ['missing_frame', 'order', 'new_birth', 'unexposed_known', 'calendar', 'history', 'summary'])
def test_whole_audit_refuses_actual_trace_scope_birth_and_update_mismatch(fault):
    videos, targets, known, window, saved, summary = synthetic_trial()
    saved, summary = deepcopy(saved), deepcopy(summary)
    if fault == 'missing_frame': saved[0]['actual_predictions'].pop()
    elif fault == 'order': saved[0]['video_ordinal_order'] = []
    elif fault in ('new_birth', 'unexposed_known'):
        event = next(event for frame in saved[0]['actual_predictions'] for event in frame['events'])
        event.update(action='NEW' if fault == 'new_birth' else 'KNOWN', category_id=None if fault == 'new_birth' else 999,
                     token='S:999' if fault == 'new_birth' else None, ambiguity_reason=None)
    elif fault == 'calendar': summary['loss_curve'][0]['actual_optimizer_family_groups'] = {}
    elif fault == 'history': saved[0]['anonymous_commit_labels'].append({'fabricated': True})
    else: summary['episodes'][0]['physical_observations'] -= 1
    with pytest.raises(ValueError):
        audit_recorded_C_trial(videos, targets, known, window, 'Top8_conditioned', 1027, saved, summary, lambda: None)


def test_reconstruction_does_not_mutate_inputs_or_call_score_model():
    videos, targets, known, window, saved, summary = synthetic_trial()
    before = json.dumps(saved, sort_keys=True), json.dumps(summary, sort_keys=True)
    audit_recorded_C_trial(videos, targets, known, window, 'Top8_conditioned', 1027, saved, summary, lambda: None)
    assert before == (json.dumps(saved, sort_keys=True), json.dumps(summary, sort_keys=True))
