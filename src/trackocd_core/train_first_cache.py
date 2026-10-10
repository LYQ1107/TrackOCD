"""Independent main Train-only cache, never silently promote old GT pilot."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from src.trackocd_v2.io import sha256_file
from src.trackocd_core.features import prefix_view


class TrainFirstCache:
    def __init__(self, root):
        import pyarrow.parquet as pq
        root=Path(root); self.manifest=json.loads((root/'manifest.json').read_text())
        done=json.loads((root/'.done').read_text())
        if self.manifest['source_role']!='TAO Train Known GT main training, not predicted/end-to-end':
            raise ValueError('Not independent Train-first main features')
        if done['manifest_sha256']!=sha256_file(root/'manifest.json'): raise ValueError('Incomplete cache')
        for name,record in self.manifest['payloads'].items():
            p=root/name
            if p.stat().st_size!=record['bytes'] or sha256_file(p)!=record['sha256']: raise ValueError('Cache payload changed')
        self.rows=pq.read_table(root/'index.parquet').to_pylist()
        self._rows={r['key']:r for r in self.rows}
        self._visual=np.load(root/'observations.npy',mmap_mode='r',allow_pickle=False)
        self._geometry=np.load(root/'geometry.npy',mmap_mode='r',allow_pickle=False)
        if self._visual.shape!=(self.manifest['observations'],768) or self._visual.dtype!=np.float16:
            raise ValueError('Wrong common features')
        if len(self.rows)!=len(self._rows) or len(self.rows)!=self.manifest['tracks']: raise ValueError('Wrong track universe')

    def get_prefix(self,key,observed):
        row=self._rows[key]; n=min(observed,row['observation_count']); i=row['observation_offset']
        return prefix_view(self._visual[i:i+n],self._geometry[i:i+n],row['quality'][:n],row['frame_ids'][:n],n)
