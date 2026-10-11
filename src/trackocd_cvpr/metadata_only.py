"""Decode scalar allowlisted metadata, not whole image/video objects or GT."""
import json
import mmap
import re

_NUMBER = re.compile(rb'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')


def _space(data, position):
    while position < len(data) and data[position] in b' \r\n\t':
        position += 1
    return position


def _end_string(data, start):
    position = start + 1
    while position < len(data):
        char = data[position]
        if char == ord('"'):
            return position + 1
        if char < 32:
            raise ValueError('Control byte in JSON string')
        if char == ord('\\'):
            position += 1
            if position == len(data):
                break
            if data[position] == ord('u'):
                digits = data[position + 1:position + 5]
                if len(digits) != 4 or any(c not in b'0123456789abcdefABCDEF' for c in digits):
                    raise ValueError('Invalid JSON unicode escape')
                position += 4
            elif data[position] not in b'"\\/bfnrt':
                raise ValueError('Invalid JSON escape')
        position += 1
    raise ValueError('Truncated JSON string')


def _end_value(data, start, depth=0):
    """Validate/skip JSON syntax without decoding any skipped value."""
    if start >= len(data) or depth > 64:
        raise ValueError('Truncated or excessively nested metadata JSON')
    opening = data[start]
    if opening == ord('"'):
        return _end_string(data, start)
    if opening not in b'[{':
        for literal in (b'true', b'false', b'null'):
            if data[start:start + len(literal)] == literal:
                return start + len(literal)
        number = _NUMBER.match(data, start)
        if number is None:
            raise ValueError('Invalid JSON scalar')
        return number.end()
    close = ord('}') if opening == ord('{') else ord(']')
    position = _space(data, start + 1)
    if position < len(data) and data[position] == close:
        return position + 1
    while position < len(data):
        if opening == ord('{'):
            if data[position] != ord('"'):
                raise ValueError('Expected JSON object key')
            position = _space(data, _end_string(data, position))
            if position == len(data) or data[position] != ord(':'):
                raise ValueError('Expected JSON key-value separator')
            position = _space(data, position + 1)
        position = _space(data, _end_value(data, position, depth + 1))
        if position < len(data) and data[position] == close:
            return position + 1
        if position == len(data) or data[position] != ord(','):
            raise ValueError('Expected JSON value separator or matching close')
        position = _space(data, position + 1)
    raise ValueError('Truncated metadata JSON')


def _project_object(data, start, end, allowed_fields):
    """Decode scalar allowlist fields, never whole objects or unknown values."""
    if data[start] != ord('{'):
        raise ValueError('Metadata array elements must be objects')
    result, seen, position = {}, set(), start + 1
    while position < end:
        position = _space(data, position)
        if data[position] == ord('}'):
            break
        if data[position] != ord('"'):
            raise ValueError('Metadata object key required')
        key_end = _end_value(data, position)
        key = json.loads(data[position:key_end])
        if key in seen:
            raise ValueError('Duplicate metadata object key')
        seen.add(key)
        position = key_end
        while position < end and chr(data[position]).isspace():
            position += 1
        if position == end or data[position] != ord(':'):
            raise ValueError('Metadata key-value separator required')
        position += 1
        while position < end and chr(data[position]).isspace():
            position += 1
        value_end = _end_value(data, position)
        if key in allowed_fields:
            if data[position] in (ord('{'), ord('[')):
                raise ValueError('Only scalar image/video metadata fields permitted')
            value = json.loads(data[position:value_end])
            expected = str if key == 'file_name' else int
            if type(value) is not expected or (expected is int and value < (1 if key in ('width', 'height') else 0)):
                raise ValueError('Invalid scalar image/video metadata type or range')
            result[key] = value
        position = _space(data, value_end)
        if position < end and data[position] == ord(','):
            position += 1
        elif position >= end or data[position] != ord('}'):
            raise ValueError('Metadata object separator required')
    return result


def _project_object_array(data, start, end, allowed_fields):
    if data[start] != ord('['):
        raise ValueError('Image/video metadata fields must be arrays')
    result, position = [], start + 1
    while position < end:
        position = _space(data, position)
        if data[position] == ord(']'):
            break
        value_end = _end_value(data, position)
        result.append(_project_object(data, position, value_end, allowed_fields))
        position = _space(data, value_end)
        if position < end and data[position] == ord(','):
            position += 1
        elif position >= end or data[position] != ord(']'):
            raise ValueError('Metadata array separator required')
    return result


def _top_level_fields(path, permitted_fields, projected_arrays=None):
    """Internal byte scanner; decode only the caller's narrow metadata fields."""
    result = {}
    with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
        p = _space(data, 0)
        if p == len(data) or data[p] != ord('{'):
            raise ValueError('Top-level metadata object required')
        p = _space(data, p + 1)
        seen = set()
        closed = False
        while p < len(data):
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
            p = _space(data, end)
            if p == len(data) or data[p] != ord(':'):
                raise ValueError('Expected key-value separator')
            p = _space(data, p + 1)
            end = _end_value(data, p)
            if key in permitted_fields:
                if projected_arrays is not None and key in projected_arrays:
                    result[key] = _project_object_array(data, p, end, projected_arrays[key])
                else:
                    result[key] = json.loads(data[p:end])
            p = _space(data, end)
            if p < len(data) and data[p] == ord(','):
                p = _space(data, p + 1)
                if p < len(data) and data[p] == ord('}'):
                    raise ValueError('Trailing top-level comma')
            elif p == len(data) or data[p] != ord('}'):
                raise ValueError('Expected top-level value separator')
        if not closed:
            raise ValueError('Truncated top-level metadata object')
    if set(result) != set(permitted_fields):
        raise ValueError('Missing permitted top-level metadata')
    return result


def image_video_metadata(path):
    """Scalar path/geometry/chronology only; nested video GT arrays skipped.

    No whole image/video objects are decoded. In particular neg_category_ids,
    not_exhaustive_category_ids, labels and arbitrary nested metadata are NOT
    permitted. Video ID is all current consumers require.
    """
    return _top_level_fields(path, ('images', 'videos'), {
        'images': ('id', 'video_id', 'frame_index', 'file_name', 'width', 'height'),
        'videos': ('id',),
    })


def train_plan_source_metadata(path):
    """No plan rows/summary/categories/heldout labels are JSON-decoded."""
    result = _top_level_fields(path, ('sources', 'annotation_sha256'))
    if not isinstance(result['sources'], dict) or not isinstance(result['annotation_sha256'], str):
        raise ValueError('Only source hashes and annotation hash metadata')
    for value in list(result['sources'].values()) + [result['annotation_sha256']]:
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Canonical SHA256 metadata required')
    return result
