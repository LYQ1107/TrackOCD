"""Read-only check of two exact COCO image ZIPs before any separate cleanup.

No dataset annotations are read; this does not authorize deletion by itself.
Images are checked against ZIP central-directory size/CRC and hashed into an
aggregate receipt. The ZIP SHA256 must match the retained original asset audit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import time
import zipfile
import zlib


ROOT = Path('/data3/liuyeqiang/pandas_bytetrack_tao_val/data/coco')
TARGETS = (
    ('train2017', 118287, '69a8bb58ea5f8f99d24875f21416de2e9ded3178e903f1f7603e283b9e06d929'),
    ('val2017', 5000, '4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05'),
)


def signature(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError('Not an owned regular file: ' + str(path))
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def verify(split, expected_count, expected_sha):
    archive = ROOT / 'downloads' / (split + '.zip')
    images = ROOT / split
    if archive.resolve() != archive or images.resolve() != images:
        raise ValueError('Unexpected symlink/path resolution')
    before = signature(archive)
    archive_hash = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            archive_hash.update(chunk)
    if archive_hash.hexdigest() != expected_sha:
        raise ValueError('Original archive SHA256 differs')
    aggregate = hashlib.sha256()
    checked = set()
    image_bytes = 0
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            if entry.is_dir():
                if entry.filename != split + '/':
                    raise ValueError('Unexpected archive directory')
                continue
            if not re.fullmatch(split + r'/[0-9]{12}\.jpg', entry.filename):
                raise ValueError('Unexpected archive entry')
            if entry.filename in checked:
                raise ValueError('Duplicate archive entry')
            checked.add(entry.filename)
            path = ROOT / entry.filename
            initial = signature(path)
            if initial[2] != entry.file_size:
                raise ValueError('Extracted size mismatch: ' + entry.filename)
            crc = 0
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    crc = zlib.crc32(chunk, crc)
                    digest.update(chunk)
            if signature(path) != initial or crc != entry.CRC:
                raise ValueError('Changed file or CRC mismatch: ' + entry.filename)
            image_bytes += entry.file_size
            aggregate.update((entry.filename + '\0' + digest.hexdigest() + '\n').encode())
    if len(checked) != expected_count or signature(archive) != before:
        raise ValueError('Archive changed or wrong extracted image count')
    return {
        'archive_path': str(archive), 'archive_bytes': before[2],
        'archive_sha256': archive_hash.hexdigest(),
        'archive_signature': [*before[:3], str(before[3])],
        'extracted_root': str(images),
        'verified_images': len(checked), 'verified_extracted_bytes': image_bytes,
        'extracted_sha256_receipt_in_zip_order': aggregate.hexdigest(),
        'verification': 'All expected image sizes/CRC match pinned original ZIP; files stable during checks. SHA256 aggregate retained; no annotation decoded.',
        'delete_authorized_by_verifier': False,
    }


if __name__ == '__main__':
    started = time.monotonic()
    results = [verify(*target) for target in TARGETS]
    print(json.dumps({
        'status': 'READ_ONLY_DUPLICATE_DOWNLOAD_VERIFICATION_PASS',
        'targets': results, 'elapsed_seconds': time.monotonic() - started,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'deleted_paths': [], 'annotations_read': False,
    }, indent=2))
