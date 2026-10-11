"""Bounded one-video reader for completed all-ID diagnostic features only."""
from pathlib import Path
import numpy as np
from src.trackocd_core.limited_masa_features import LimitedMasaVideoCache
from src.trackocd_v2.io import sha256_file


class DiagnosticFeatureCache:
    def __init__(self, root, records, guard=lambda: None):
        self.root = Path(root).resolve()
        self.guard = guard
        self.rows, self.paths, self.by_key = [], {}, {}
        self.reader = None
        self.video_id = None
        self.observations = 0
        for record in records:
            self.guard()
            video = record['video_id']
            directory, filename = Path(record['directory']), Path(record['npz_filename'])
            if (type(video) is not int or video in self.paths or directory.is_absolute()
                    or '..' in directory.parts or filename.name != str(filename)):
                raise ValueError('Unsafe or duplicate diagnostic feature reference')
            path = self.root / directory / filename
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise ValueError('Feature reference escapes registered project')
            if path.stat().st_size != record['npz_bytes'] or sha256_file(path) != record['npz_sha256']:
                raise ValueError('Completed feature payload changed')
            reader = LimitedMasaVideoCache(path, video)
            try:
                count = len(reader.data['visual'])
                if len(reader.rows) != record['tracks'] or count != record['observations']:
                    raise ValueError('All predicted IDs/observations required')
                if (reader.data['geometry'].shape != (count, 4)
                        or reader.data['quality'].shape != (count,)
                        or not np.isfinite(reader.data['geometry']).all()
                        or not np.isfinite(reader.data['quality']).all()):
                    raise ValueError('Nonfinite/malformed diagnostic observation input')
                for row in reader.rows:
                    if (row['key'] in self.by_key or any(a >= b for a, b in zip(row['frame_ids'], row['frame_ids'][1:]))):
                        raise ValueError('Duplicate identity or noncausal frame index')
                    self.rows.append(row)
                    self.by_key[row['key']] = row
                self.observations += count
                self.paths[video] = path
            finally:
                reader.close()

    def video(self, video_id):
        if self.video_id != video_id:
            self.guard()
            self.close()
            self.reader = LimitedMasaVideoCache(self.paths[video_id], video_id)
            self.video_id = video_id
        return self.reader

    def get_prefix(self, key, cap):
        if type(cap) is not int or cap not in (1, 2, 4, 8, 16):
            raise ValueError('Registered visible prefix only')
        return self.video(self.by_key[key]['video_id']).get_prefix(key, cap)

    def close(self):
        if self.reader is not None:
            self.reader.close()
        self.reader, self.video_id = None, None
