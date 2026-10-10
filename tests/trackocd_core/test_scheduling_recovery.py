import json
import os
from pathlib import Path
import socket
import threading

import pytest
from scripts.trackocd_core.resume_limited_masa_inference import snapshot, same_process, headroom, verify_group
from scripts.trackocd_core.local_proxy_relay import relay, serve


def fake_process(tmp_path, *, start='101', state='S', argv=('python', 'worker'), pages=2):
    directory = tmp_path / '42'
    directory.mkdir(exist_ok=True)
    fields = [state] + ['0'] * 18 + [start] + ['0'] * 3
    (directory / 'stat').write_text('42 (worker (name)) ' + ' '.join(fields))
    (directory / 'cmdline').write_bytes('\0'.join(argv).encode() + b'\0')
    (directory / 'statm').write_text(f'100 {pages} 0')
    return snapshot(42, tmp_path)


def test_recovery_uses_current_start_uid_and_exact_command(tmp_path):
    witness = fake_process(tmp_path)
    assert same_process(witness, tmp_path)
    fake_process(tmp_path, start='102')
    assert not same_process(witness, tmp_path)  # PID reuse cannot confer ownership.
    witness = fake_process(tmp_path)
    fake_process(tmp_path, argv=('python', 'foreign'))
    assert not same_process(witness, tmp_path)
    witness = fake_process(tmp_path)
    fake_process(tmp_path, state='Z')
    assert not same_process(witness, tmp_path)
    assert snapshot(99, tmp_path) is None


def test_joint_reservation_is_unallocated_peak_not_double_counted_rss(tmp_path):
    witness = fake_process(tmp_path)
    page = os.sysconf('SC_PAGE_SIZE')
    assert headroom([witness], page * 5, tmp_path) == page * 3
    assert headroom([witness], page, tmp_path) == 0
    fake_process(tmp_path, start='999')
    assert headroom([witness], page * 5, tmp_path) == 0


def test_orphan_needs_atomic_terminal_and_every_immutable_seal(tmp_path):
    plan = {'groups': {'group': ['a', 'b']}, 'jobs': {'a': {'id': 'a'}, 'b': {'id': 'b'}}}
    path = tmp_path / 'group_attempt_done.json'
    with pytest.raises(FileNotFoundError):
        verify_group(tmp_path, plan, 'group', {}, 'attempt', lambda *a: {})
    path.write_text(json.dumps({'group': 'group', 'completed': ['a', 'a']}))
    with pytest.raises(ValueError, match='terminal receipt'):
        verify_group(tmp_path, plan, 'group', {}, 'attempt', lambda *a: {})
    path.write_text(json.dumps({'group': 'group', 'completed': ['a', 'b']}))
    with pytest.raises(ValueError, match='actually be sealed'):
        verify_group(tmp_path, plan, 'group', {}, 'attempt', lambda *a: None)
    assert verify_group(tmp_path, plan, 'group', {}, 'attempt', lambda *a: {})['completed'] == ['a', 'b']


def test_loopback_relay_preserves_opaque_bytes_and_half_close():
    with socket.socket() as upstream:
        upstream.bind(('127.0.0.1', 0))
        upstream.listen(1)
        port = upstream.getsockname()[1]
        errors = []

        def echo():
            try:
                with upstream.accept()[0] as connection:
                    connection.settimeout(3)
                    parts = []
                    while True:
                        data = connection.recv(65536)
                        if not data:
                            break
                        parts.append(data)
                    connection.sendall(b''.join(parts) + b'-reply')
            except Exception as exc:
                errors.append(exc)

        echo_thread = threading.Thread(target=echo, daemon=True)
        echo_thread.start()
        client, accepted = socket.socketpair()
        with client:
            relay_thread = threading.Thread(target=relay, args=(accepted, port), daemon=True)
            relay_thread.start()
            client.settimeout(3)
            client.sendall(b'opaque\x00bytes')
            client.shutdown(socket.SHUT_WR)
            assert client.recv(65536) == b'opaque\x00bytes-reply'
            assert client.recv(1) == b''
        relay_thread.join(3)
        echo_thread.join(3)
        assert not relay_thread.is_alive() and not echo_thread.is_alive() and not errors


@pytest.mark.parametrize('listen,upstream', [(17890,17890), (0,17899), (17890,65536)])
def test_invalid_or_recursive_relay_rejected_before_binding(listen, upstream):
    with pytest.raises(ValueError):
        serve(listen, upstream)


def test_safety_cleanup_never_includes_adopted_workers():
    source = Path('scripts/trackocd_core/resume_limited_masa_inference.py').read_text()
    cleanup = source.split('except Exception as exc:', 1)[1].split('finally:', 1)[0]
    assert 'for child in children:' in cleanup
    assert "entry['witness']['pid']" not in cleanup
    assert 'models_checkpoints_operating_points_plan_or_frozen_executors_modified' in source
    assert 'UNAVAILABLE_NOT_A_CHILD' in source
    assert 'start_new_session=True' in source
