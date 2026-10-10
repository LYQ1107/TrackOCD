"""One frozen representation bank in RAM; common features streamed by video.

No persistent projected cache, GT matches/targets or semantic vocabulary.
Learned forward uses the identical registered CPU FP32 helper and batch64;
~1.5GiB per 256D family, not every family/all raw frame copies in memory.
"""
from __future__ import annotations
import hashlib
import numpy as np
from src.trackocd_core.limited_masa_features import LimitedMasaVideoCache
from src.trackocd_core.train_first_experiment import evidence_bank
from src.trackocd_v2.io import sha256_file


class FrozenVideoBank:
    def __init__(self,directory,records,model=None,no_reliability=False,progress=None):
        self.directory=directory;self.records={r['video_id']:r for r in records};self.model=model
        self._verified={}
        self.vectors=None;self.scalars=None;self.starts={};self.cache=None;self.cache_video=None
        total=sum(r['tracks'] for r in records)
        if model is not None:
            self.vectors=np.empty((5,total,256),np.float32);self.scalars=np.empty((5,total,2),np.float32)
            offset=0
            for record in records:
                self.starts[record['video_id']]=offset
                cache=self._open(record['video_id'])
                bank=evidence_bank(cache,cache.rows,model,no_reliability)
                for j,cap in enumerate((1,2,4,8,16)):
                    for i,r in enumerate(cache.rows):
                        item=bank[cap][r['key']];self.vectors[j,offset+i]=item['embedding']
                        self.scalars[j,offset+i]=[item['uncertainty'],item['maturity']]
                offset+=len(cache.rows);cache.close();del bank
                if progress:progress(record['video_id'],offset)
            if offset!=total or not np.isfinite(self.vectors).all() or not np.isfinite(self.scalars).all():raise ValueError('Incomplete frozen bank')
            self.vectors.flags.writeable=False;self.scalars.flags.writeable=False
        self.sha256=(hashlib.sha256(memoryview(self.vectors).cast('B')).hexdigest() if self.vectors is not None else None)
        self.scalar_sha256=(hashlib.sha256(memoryview(self.scalars).cast('B')).hexdigest() if self.scalars is not None else None)

    def _open(self,video):
        record=self.records[video];path=self.directory/record['npz_filename'];before=path.stat()
        signature=(before.st_size,before.st_mtime_ns)
        if path.is_symlink() or before.st_size!=record['npz_bytes']:raise ValueError('Frozen feature path/size changed')
        if video in self._verified:
            if self._verified[video]!=signature:raise ValueError('Feature changed after verification; do not refresh silently')
        elif sha256_file(path)!=record['npz_sha256']:raise ValueError('Frozen feature payload changed')
        cache=LimitedMasaVideoCache(path,video);after=path.stat()
        if (after.st_size,after.st_mtime_ns)!=signature:raise ValueError('Feature changed while reading')
        self._verified[video]=signature;return cache

    def videos(self,video,cap):
        if self.cache_video!=video:
            if self.cache:self.cache.close()
            self.cache=self._open(video);self.cache_video=video
        cache=self.cache;j=(1,2,4,8,16).index(cap);start=self.starts.get(video,0)
        indexes=sorted(range(len(cache.rows)),key=lambda i:(cache.rows[i]['frame_ids'][min(cap,cache.rows[i]['observation_count'])-1],cache.rows[i]['physical_track_id']))
        for i in indexes:
            row=cache.rows[i];v=cache.get_prefix(row['key'],cap)
            item={'quality':v.quality,'elapsed':float(v.elapsed_frames[-1]),'elapsed_frames':v.elapsed_frames,'actual_prefix':len(v.visual)}
            if self.vectors is None:
                z=v.weighted_mean();item.update(embedding=z,maturity=float(len(v.visual)),uncertainty=float(np.clip(1-(v.visual@z).mean(),0,1)),frames=v.visual)
            else:
                item.update(embedding=self.vectors[j,start+i],uncertainty=float(self.scalars[j,start+i,0]),maturity=float(self.scalars[j,start+i,1]))
            yield row,item

    def close(self):
        if self.cache:self.cache.close()
        self.cache=None;self.vectors=None;self.scalars=None
