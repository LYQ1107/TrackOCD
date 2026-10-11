"""Read top-level dataset image/video metadata without decoding label arrays."""
import json
import mmap


def _end_value(data, start):
    opening = data[start]
    if opening not in (ord('['), ord('{'), ord('"')):
        p = start
        while p < len(data) and data[p] not in (ord(','), ord('}')):
            p += 1
        return p
    depth = 0
    quoted = False
    escaped = False
    for p in range(start, len(data)):
        c = data[p]
        if quoted:
            if escaped:
                escaped = False
            elif c == ord('\\'):
                escaped = True
            elif c == ord('"'):
                quoted = False
                if opening == ord('"') and depth == 0:
                    return p + 1
        elif c == ord('"'):
            quoted = True
        elif c in (ord('['), ord('{')):
            depth += 1
        elif c in (ord(']'), ord('}')):
            depth -= 1
            if depth == 0:
                return p + 1
    raise ValueError('Truncated metadata JSON')


def image_video_metadata(path):
    """Skipped annotations/categories are byte-scanned, never JSON-decoded."""
    result = {}
    with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        p = 0
        while p < len(data) and chr(data[p]).isspace():
            p += 1
        if p == len(data) or data[p] != ord('{'):
            raise ValueError('Top-level metadata object required')
        p += 1
        seen = set()
        closed = False
        while p < len(data):
            while p < len(data) and (chr(data[p]).isspace() or data[p] == ord(',')):
                p += 1
            if p == len(data):
                raise ValueError('Truncated top-level object')
            if data[p] == ord('}'):
                closed = True
                if data[p + 1:].strip():
                    raise ValueError('Trailing metadata input')
                break
            if data[p] != ord('"'):
                raise ValueError('Expected top-level metadata key')
            end = _end_value(data, p)
            key = json.loads(data[p:end])
            if key in seen:
                raise ValueError('Duplicate top-level metadata key')
            seen.add(key)
            p = end
        if not closed:
            raise ValueError('Truncated top-level metadata object')
            while p < len(data) and chr(data[p]).isspace():
                p += 1
            if data[p] != ord(':'):
                raise ValueError('Expected key-value separator')
            p += 1
            while p < len(data) and chr(data[p]).isspace():
                p += 1
            end = _end_value(data, p)
            if key in ('images', 'videos'):
                result[key] = json.loads(data[p:end])
                if not isinstance(result[key], list):
                    raise ValueError('Metadata fields must be arrays')
            p = end
    if 'images' not in result or 'videos' not in result:
        raise ValueError('Missing image/video metadata')
    return result
