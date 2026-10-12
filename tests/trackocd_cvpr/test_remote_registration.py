import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.trackocd_cvpr.remote_registration import verify_registration, validate_ref_response, BRANCH

SHA = 'a' * 40


def ref(sha=SHA, branch=BRANCH):
    return {'ref': 'refs/heads/' + branch, 'object': {'type': 'commit', 'sha': sha}}


def test_git_success_never_requests_API(tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=(SHA + '\trefs/heads/' + BRANCH + '\n').encode())
    result = verify_registration(tmp_path, BRANCH, SHA, run=run, allow_public_api=True)
    assert len(calls) == 1 and result['method'] == 'fresh_git_ls_remote'


def test_valid_git_mismatch_must_not_fallback(tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=('b' * 40 + '\trefs/heads/' + BRANCH).encode())
    with pytest.raises(ValueError):
        verify_registration(tmp_path, BRANCH, SHA, run=run, allow_public_api=True)
    assert len(calls) == 1


def test_transport_failure_gets_only_exact_https_public_ref_without_credentials(tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == 'git':
            return SimpleNamespace(returncode=128, stdout=b'')
        assert command[0] == 'curl' and '--insecure' not in command
        assert command[-1] == 'https://api.github.com/repos/LYQ1107/TrackOCD/git/ref/heads/' + BRANCH
        assert not any('Authorization' in arg for arg in command)
        return SimpleNamespace(returncode=0, stdout=json.dumps(ref()).encode())
    result = verify_registration(tmp_path, BRANCH, SHA, run=run, allow_public_api=True)
    assert len(calls) == 2
    assert result['method'] == 'fresh_public_GitHub_exact_ref_API'
    assert result['credentials_or_local_receipts_used_for_API_proof'] is False


@pytest.mark.parametrize('value', [[], ref('b' * 40), ref(SHA, BRANCH + '-suffix'),
                                  {'ref': 'refs/heads/' + BRANCH, 'object': {'type': 'tag', 'sha': SHA}}])
def test_API_shape_sha_prefix_branch_or_object_mutations_refused(value):
    with pytest.raises(ValueError):
        validate_ref_response(value, SHA)


def test_both_transports_fail_bounded_and_gate_stays_closed(tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=35, stdout=b'')
    with pytest.raises(RuntimeError, match='gate remains closed'):
        verify_registration(tmp_path, BRANCH, SHA, run=run, allow_public_api=True)
    assert len(calls) == 3
    calls.clear()
    with pytest.raises(RuntimeError, match='not registered'):
        verify_registration(tmp_path, BRANCH, SHA, run=run)
    assert len(calls) == 1


def test_v3_does_not_change_v2_data_or_structural_boundaries():
    root = Path(__file__).resolve().parents[2]
    a = json.loads((root / 'configs/trackocd_cvpr/joint_known_protocol_v2_unbalanced.json').read_text())
    b = json.loads((root / 'configs/trackocd_cvpr/joint_known_protocol_v3_unbalanced_ref_transport.json').read_text())
    for key, value in a.items():
        if key not in ('schema_version', 'output_directory', 'interpretation'):
            assert b[key] == value
