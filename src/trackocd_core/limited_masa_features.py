"""All-ID first-observed-prefix planning and atomic compact feature readers."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from src.trackocd_core.features import prefix_view
from src.trackocd_v2.io import sha256_file


def select_all_prefixes(video,arrays,cap=16):
    """No GT/future length/high-score sampling; all first occurrences kept."""
    grouped={};first={}
    for i,image in enumerate(video['images']):
        begin,end=map(int,arrays['frame_offsets'][i:i+2])
        for j in range(begin,end):
            identity=int(arrays['track_id'][j])
            if identity not in grouped:grouped[identity]=[];first[identity]=i
            if len(grouped[identity])<cap:
                grouped[identity].append({'image_position':i,'source_offset':j})
    rows=[];offset=0
    for identity in sorted(grouped,key=lambda t:(first[t],t)):
        observations=grouped[identity]
        rows.append({'physical_track_id':identity,'observation_offset':offset,'observations':observations});offset+=len(observations)
    if offset>cap*len(rows):raise AssertionError('Prefix ceiling violated')
    return rows,offset


def completed_feature(directory,video_id,config_sha256,input_sha256):
    marker=Path(directory)/f'video_{video_id:04d}.complete.json'
    if not marker.exists():return None
    if marker.is_symlink():raise ValueError('Unexpected marker link')
    record=json.loads(marker.read_text());filename=Path(record['npz_filename']);path=marker.parent/filename
    if (filename.name!=str(filename) or path.is_symlink() or record['status']!='SEALED_COMPLETE_ALL_ID_PREFIX_FEATURES'
        or record['video_id']!=video_id or record['config_sha256']!=config_sha256 or record['input_npz_sha256']!=input_sha256
        or path.stat().st_size!=record['npz_bytes'] or sha256_file(path)!=record['npz_sha256']):raise ValueError('Invalid completed feature shard,never overwrite')
    return record


class LimitedMasaVideoCache:
    def __init__(self,path,video_id):
        with np.load(path,allow_pickle=False) as payload:self.data={k:payload[k] for k in payload.files}
        a=self.data;self.video_id=video_id
        if a['visual'].shape!=(int(a['offsets'][-1]),768) or a['visual'].dtype!=np.float16 or not np.isfinite(a['visual']).all():raise ValueError('Invalid compact predicted features')
        ids=a['track_id'].tolist();offsets=a['offsets']
        if len(set(ids))!=len(ids) or offsets[0]!=0 or np.any(np.diff(offsets)<1) or np.any(np.diff(offsets)>16):raise ValueError('Invalid predicted index')
        self.rows=[];self._rows={}
        for i,identity in enumerate(ids):
            begin,end=map(int,offsets[i:i+2]);key=f'{video_id}:{identity}'
            row={'key':key,'video_id':video_id,'physical_track_id':int(identity),'observation_count':end-begin,'observation_offset':begin,
                 'frame_ids':a['frame_index'][begin:end].tolist()}
            self.rows.append(row);self._rows[key]=row
    def get_prefix(self,key,cap):
        r=self._rows[key];n=min(cap,r['observation_count']);i=r['observation_offset'];a=self.data
        return prefix_view(a['visual'][i:i+n],a['geometry'][i:i+n],a['quality'][i:i+n],a['frame_index'][i:i+n],n)
    def close(self):self.data.clear()
