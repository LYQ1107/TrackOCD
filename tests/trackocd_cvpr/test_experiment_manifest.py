import hashlib
from unittest.mock import patch
import pytest
from scripts.trackocd_cvpr.write_cvpr0_experiment_manifest import verified_sources


def test_manifest_receipt_actual_git_bytes_not_current_source_assumption(tmp_path):
    digest = hashlib.sha256(b'old frozen source').hexdigest()
    with patch('scripts.trackocd_cvpr.write_cvpr0_experiment_manifest.subprocess.check_output', return_value=b'old frozen source'):
        assert verified_sources(tmp_path, 'a' * 40, {'source.py': digest})
    with patch('scripts.trackocd_cvpr.write_cvpr0_experiment_manifest.subprocess.check_output', return_value=b'changed source'):
        with pytest.raises(ValueError):
            verified_sources(tmp_path, 'a' * 40, {'source.py': digest})
    with pytest.raises(ValueError):
        verified_sources(tmp_path, 'aaaaaaa', {'source.py': digest})
