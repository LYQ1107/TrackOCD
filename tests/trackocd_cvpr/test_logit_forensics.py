import pytest
from src.trackocd_cvpr.logit_forensics import LogitForensics


def test_exact_bias_and_unavailable_action_not_in_logit_means():
    a = LogitForensics()
    features = [0.] * 17
    features[15] = 1.
    a.observe([0., -1e9, -1., 1.], features, 0, 'known', 1, None,
              [0., 10., 4., .55], 0, 1, -2., 0, None)
    r = a.finish()
    assert r['logit_channels']['EXISTING:raw']['mean'] is None
    assert r['logit_channels']['EXISTING:raw']['masked'] == 1
    assert r['logit_channels']['WAIT:raw']['mean'] == 1.
    assert r['logit_channels']['WAIT:effective_WAIT_bias']['mean'] == -1.
    assert r['counts']['wrong_known'] == 0


def test_available_correct_top1_and_unnecessary_wait():
    a = LogitForensics()
    features = [0.] * 17
    features[15] = 1.
    a.observe([0., -1e9, -1., 1.], features, 3, 'known', 1, None,
              [0., 10., 4., .55], 0, 1, 0., 0, None)
    r = a.finish()
    assert r['counts']['known_correct_exposed_top1_but_wrong_action'] == 1
    assert r['counts']['unnecessary_WAIT_under_nominal_current_top1_target'] == 1
    assert r['premature_commit_causality'].startswith('NOT_ESTABLISHED')


def test_invalid_retained_action_rejected():
    a = LogitForensics()
    with pytest.raises(ValueError, match='Retained action'):
        a.observe([0., -1e9, -1., 1.], [0.] * 17, 0, 'known', 1, None,
                  [0., 10., 4., .55], 0, 1, 0., 0, None)
