"""Equivalent fresh exact-ref verification, never a local receipt bypass."""
import json
import os
import re
import subprocess

BRANCH = 'research/cvpr-trackocd-autonomous-v2'
REPO = 'LYQ1107/TrackOCD'
PUBLIC_API_ATTEMPTS = 8


def validate_ref_response(value, expected):
    if (not isinstance(value, dict) or value.get('ref') != 'refs/heads/' + BRANCH
            or not isinstance(value.get('object'), dict) or value['object'].get('type') != 'commit'
            or value['object'].get('sha') != expected):
        raise ValueError('Exact live branch and full expected commit required, no matching-prefix or cached local receipt')
    return True


def verify_registration(root, branch, expected, proxy='', *, allow_public_api=False, run=subprocess.run):
    if branch != BRANCH or re.fullmatch('[0-9a-f]{40}', expected) is None:
        raise ValueError('Only this exact repository research branch and full Git SHA')
    git = ['git'] + (['-c', 'http.proxy=' + proxy] if proxy else []) + ['-c', 'http.version=HTTP/1.1']
    error_kind, error_code = None, None
    try:
        response = run(git + ['ls-remote', 'origin', 'refs/heads/' + branch], cwd=root,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=35,
                       env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
        if response.returncode == 0:
            rows = response.stdout.decode().strip().splitlines()
            if len(rows) != 1 or rows[0].split() != [expected, 'refs/heads/' + branch]:
                raise ValueError('Remote branch differs or missing; do not fallback after a valid mismatched Git response')
            return {'method': 'fresh_git_ls_remote', 'ref': 'refs/heads/' + branch, 'sha': expected}
        error_kind, error_code = 'git_transport_failure', response.returncode
    except subprocess.TimeoutExpired:
        error_kind = 'git_transport_timeout'
    if not allow_public_api:
        raise RuntimeError('Git registration transport failed; API fallback not registered for this config')
    url = 'https://api.github.com/repos/' + REPO + '/git/ref/heads/' + branch
    command = ['curl', '--fail', '--silent', '--show-error', '--http1.1', '--tlsv1.2',
               '--proto', '=https',
               '--connect-timeout', '6', '--max-time', '12', '--max-filesize', '65536',
               '--header', 'Accept: application/vnd.github+json', '--header', 'Cache-Control: no-cache',
               '--header', 'X-GitHub-Api-Version: 2022-11-28']
    if proxy:
        command += ['--proxy', proxy]
    command.append(url)
    for attempt in range(1, PUBLIC_API_ATTEMPTS + 1):
        try:
            response = run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        except subprocess.TimeoutExpired:
            continue
        if response.returncode != 0:
            continue
        if len(response.stdout) > 65536:
            raise ValueError('Small ref metadata bound')
        value = json.loads(response.stdout)
        validate_ref_response(value, expected)
        return {'method': 'fresh_public_GitHub_exact_ref_API', 'ref': value['ref'], 'sha': expected,
                'canonical_API_url': url, 'API_attempt': attempt,
                'primary_git_transport_error': error_kind, 'primary_git_exit_code': error_code,
                'credentials_or_local_receipts_used_for_API_proof': False}
    raise RuntimeError('All bounded live remote-ref transports failed; private input gate remains closed')
