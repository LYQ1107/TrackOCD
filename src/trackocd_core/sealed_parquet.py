"""Compact private ledger, no GT fields; round-trippable audited sealing."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from src.trackocd_core.evaluation import DecisionEvent,TrackKey,seal_decisions
KINDS=('KNOWN','NEW','EXISTING','WAIT')


def write_sealed(path,replay):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path=Path(path);temporary=path.with_suffix('.parquet.tmp')
    if path.exists() or temporary.exists():raise ValueError('Preserve retained actual sealed ledger')
    styles={e.token.split(':')[0] for e in replay.events if e.token is not None}
    if len(styles)>1 or styles-set(('A','S')):raise ValueError('Unexpected anonymous namespace')
    namespace=next(iter(styles),'A')
    n=len(replay.events);video=np.empty(n,np.int32);physical=np.empty(n,np.int64);observed=np.empty(n,np.uint8)
    kind=np.empty(n,np.uint8);known=np.empty(n,np.int32);token=np.empty(n,np.int32)
    for i,e in enumerate(replay.events):
        if e.sequence!=i:raise ValueError('Expected contiguous immutable sequence')
        video[i]=e.physical_key.video_id;physical[i]=int(e.physical_key.local_track_id);observed[i]=e.observed_prefix
        kind[i]=KINDS.index(e.kind);known[i]=e.known_category_id if e.known_category_id is not None else -1;token[i]=int(e.token.split(':')[1]) if e.token else -1
    table=pa.table({'video':video,'physical':physical,'observed':observed,'kind':kind,'known':known,'token':token})
    table=table.replace_schema_metadata({b'namespace':namespace.encode(),b'prefix':str(replay.prefix_cap).encode()})
    pq.write_table(table,temporary,compression='zstd')
    with temporary.open('rb') as reader:os.fsync(reader.fileno())
    temporary.rename(path)


def read_sealed(path,order,known_ids):
    import pyarrow.parquet as pq
    t=pq.read_table(path);meta=t.schema.metadata;a={k:t[k].to_numpy() for k in t.column_names};style=meta[b'namespace'].decode()
    events=[]
    for i in range(len(t)):
        kind=KINDS[int(a['kind'][i])]
        events.append(DecisionEvent(i,TrackKey(int(a['video'][i]),str(int(a['physical'][i]))),int(a['observed'][i]),kind,
            known_category_id=int(a['known'][i]) if kind=='KNOWN' else None,
            token=f"{style}:{int(a['token'][i])}" if kind in {'NEW','EXISTING'} else None))
    return seal_decisions(events,video_order=order,prefix_cap=int(meta[b'prefix']),known_ids=known_ids)
