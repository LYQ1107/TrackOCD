"""Causal canonical-root reconstruction for the Phase86 RF diagnostic.

The union graph is replayed only up to the anchor observation.  It is an
offline representation audit; fragment IDs never enter a model tensor.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any
import json
import numpy as np
from src.iclr27_phase23.protocol import order_key, track_key
from scripts.iclr27_phase85.build_physical_r_adapter import join_rows

@dataclass(frozen=True)
class UnionEvent:
    video_id: int
    step: int
    child_id: int
    parent_id: int

class DisjointSet:
    def __init__(self): self.parent: dict[int,int] = {}
    def find(self, x: int) -> int:
        if x not in self.parent: self.parent[x] = x
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.parent[rb] = ra

class CausalUnionTimeline:
    def __init__(self, improved_rows: list[dict[str,Any]], union_events: list[UnionEvent]):
        self.rows_by_video: dict[int,list[int]] = defaultdict(list)
        self.events_by_video: dict[int,list[UnionEvent]] = defaultdict(list)
        self.ids_by_video: dict[int,set[int]] = defaultdict(set)
        for i, row in enumerate(improved_rows):
            v=int(row.get('video_id',-1)); oid=int(row.get('original_physical_track_id',-1)); self.rows_by_video[v].append(i); self.ids_by_video[v].add(oid)
        for event in union_events:
            self.events_by_video[event.video_id].append(event); self.ids_by_video[event.video_id].update((event.child_id,event.parent_id))
        for v in self.rows_by_video: self.rows_by_video[v].sort(key=lambda i:(int(improved_rows[i].get('frame_id',0)),int(improved_rows[i].get('image_id',0)),i))
        for v in self.events_by_video: self.events_by_video[v].sort(key=lambda e:(e.step,e.child_id,e.parent_id))
    def members_at(self, video_id: int, cutoff_step: int, anchor_original_id: int) -> set[int]:
        d=DisjointSet()
        for x in self.ids_by_video.get(video_id,set()): d.find(x)
        for e in self.events_by_video.get(video_id,[]):
            if e.step > cutoff_step: break
            d.union(e.child_id,e.parent_id)
        root=d.find(anchor_original_id)
        return {x for x in self.ids_by_video.get(video_id,set()) if d.find(x)==root}

class FragmentSetBuilder:
    def __init__(self, table, public_rows, improved_native_rows, timeline: CausalUnionTimeline, mapped: np.ndarray, mapped_iou: np.ndarray):
        self.table=table; self.public_rows=public_rows; self.native=improved_native_rows; self.timeline=timeline; self.mapped=mapped; self.mapped_iou=mapped_iou
        self.by_track=defaultdict(list); self.native_to_public=defaultdict(list)
        for i,row in enumerate(public_rows): self.by_track[track_key(row)].append(i)
        for k in self.by_track: self.by_track[k].sort(key=lambda i:order_key(public_rows[i]))
        for i,n in enumerate(mapped):
            if n>=0 and mapped_iou[i]>=.5: self.native_to_public[int(n)].append(i)
        self.original_rows=defaultdict(list)
        for n, ids in self.native_to_public.items():
            oid=int(improved_native_rows[n].get('original_physical_track_id',-1)); self.original_rows[oid].extend(ids)
    def build(self, track_key_value: str, prefix: int) -> dict[str,Any]:
        seq=self.by_track.get(track_key_value,[]); use=seq[:min(int(prefix),len(seq))]; good=[i for i in use if self.mapped[i]>=0 and self.mapped_iou[i]>=.5]
        raw=self.table.raw_vector(track_key_value,prefix)
        if not good: return {'track_key':track_key_value,'prefix':int(prefix),'video_id':int(self.table.metadata[track_key_value]['video']),'anchor_original_id':None,'cutoff_step':None,'fragment_ids':[],'fragment_vectors':np.zeros((0,768),np.float32),'raw_anchor_vector':raw,'fallback':True}
        anchor_public=good[-1]; nrow=self.native[int(self.mapped[anchor_public])]; video=int(nrow.get('video_id',-1)); cutoff=int(nrow.get('image_id',nrow.get('frame_id',0))); anchor=int(nrow.get('original_physical_track_id',-1)); members=sorted(self.timeline.members_at(video,cutoff,anchor)); vectors=[]; present=[]
        for oid in members:
            idx=[i for i in self.original_rows.get(oid,[]) if int(self.public_rows[i].get('video_id',-1))==video and int(self.public_rows[i].get('image_id',self.public_rows[i].get('frame_id',0)))<=cutoff]
            if not idx: continue
            arr=self.table.features[np.asarray(idx,dtype=np.int64)]; arr=arr/np.maximum(np.linalg.norm(arr,axis=1,keepdims=True),1e-8); v=arr.mean(axis=0); v=v/max(float(np.linalg.norm(v)),1e-8); vectors.append(v.astype(np.float32)); present.append(oid)
        return {'track_key':track_key_value,'prefix':int(prefix),'video_id':video,'anchor_original_id':anchor,'cutoff_step':cutoff,'fragment_ids':present,'fragment_vectors':np.asarray(vectors,np.float32).reshape((-1,768)) if vectors else np.zeros((0,768),np.float32),'raw_anchor_vector':raw,'fallback':not bool(vectors)}

def load_timeline(lineage_path: Path, union_path: Path):
    lineage=[json.loads(line) for line in lineage_path.open() if line.strip()]
    events=[]
    for line in union_path.open():
        if not line.strip(): continue
        obj=json.loads(line); events.append(UnionEvent(int(obj['video_id']),int(obj.get('step',obj.get('image_id',0))),int(obj['child_original_physical_track_id']),int(obj['parent_canonical_physical_track_id'])))
    return lineage, CausalUnionTimeline(lineage,events)

def make_builder(table, public_rows, lineage_path: Path, union_path: Path):
    lineage,timeline=load_timeline(lineage_path,union_path); mapped,miou,_,audit=join_rows(public_rows,lineage); return FragmentSetBuilder(table,public_rows,lineage,timeline,mapped,miou), {'lineage_rows':len(lineage),'union_events':sum(len(x) for x in timeline.events_by_video.values()),'join':audit}
