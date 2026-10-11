import hashlib
import zipfile

import pytest

from scripts.trackocd_cvpr import verify_disposable_coco_archives as audit


def fixture(tmp_path, monkeypatch, entry='train2017/000000000001.jpg'):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    (tmp_path / 'downloads').mkdir()
    (tmp_path / 'train2017').mkdir()
    archive = tmp_path / 'downloads/train2017.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr(entry, b'synthetic-image')
    (tmp_path / 'train2017/000000000001.jpg').write_bytes(b'synthetic-image')
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def test_all_checks_pass_but_verifier_never_deletes(tmp_path, monkeypatch):
    archive, digest = fixture(tmp_path, monkeypatch)
    result = audit.verify('train2017', 1, digest)
    assert result['verified_images'] == 1
    assert result['archive_sha256'] == digest
    assert not result['delete_authorized_by_verifier']
    assert archive.exists()
    assert (tmp_path / 'train2017/000000000001.jpg').exists()


@pytest.mark.parametrize('damage', ['crc', 'size', 'missing', 'symlink', 'count', 'hash'])
def test_incomplete_or_changed_extraction_is_rejected(tmp_path, monkeypatch, damage):
    archive, digest = fixture(tmp_path, monkeypatch)
    image = tmp_path / 'train2017/000000000001.jpg'
    count = 1
    if damage == 'crc':
        image.write_bytes(b'Synthetic-image')
    elif damage == 'size':
        image.write_bytes(b'wrong')
    elif damage == 'missing':
        image.unlink()
    elif damage == 'symlink':
        image.unlink()
        image.symlink_to(archive)
    elif damage == 'count':
        count = 2
    elif damage == 'hash':
        digest = '0' * 64
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.verify('train2017', count, digest)
    assert archive.exists()


def test_nonimage_or_path_escape_entry_is_rejected(tmp_path, monkeypatch):
    archive, digest = fixture(tmp_path, monkeypatch, entry='../annotations.json')
    with pytest.raises(ValueError, match='Unexpected archive entry'):
        audit.verify('train2017', 1, digest)
    assert archive.exists()
