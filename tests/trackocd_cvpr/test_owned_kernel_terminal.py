from scripts.trackocd_cvpr.audit_joint_known_protocol import kernel_terminal_status


def test_kernel_dead_or_zombie_only_not_running_sleeping_or_missing():
    assert kernel_terminal_status('Name:\tworker\nState:\tZ (zombie)\nThreads:\t1\n')
    assert kernel_terminal_status('State:\tX (dead)\n')
    assert kernel_terminal_status('State:\tx (dead)\n')
    for state in ('R (running)', 'S (sleeping)', 'D (disk sleep)', 'T (stopped)', 'I (idle)'):
        assert not kernel_terminal_status('State:\t' + state + '\n')
    assert not kernel_terminal_status('Name:\tZ\nRss:\t0\n')
    assert not kernel_terminal_status('State:\t\n')
