"""Small raw-RGB causal TrackOCD model used for Phase60.

This implementation deliberately has no category, text, physical-ID or
semantic-ID inputs.  The detector is a dense class-agnostic grid head; the
track/semantic/controller heads consume only visual activations and causal
metadata.  It is intentionally compact so that the four-fold experiment is
reproducible on four GPUs.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class PixelTrackOCD(nn.Module):
    def __init__(self, image_size: int = 224):
        super().__init__()
        self.image_size=image_size
        self.backbone=nn.Sequential(
            nn.Conv2d(3,32,5,2,2), nn.GroupNorm(8,32), nn.GELU(),
            nn.Conv2d(32,64,3,2,1), nn.GroupNorm(8,64), nn.GELU(),
            nn.Conv2d(64,128,3,2,1), nn.GroupNorm(16,128), nn.GELU(),
            nn.Conv2d(128,128,3,1,1), nn.GroupNorm(16,128), nn.GELU(),
        )
        # Absolute normalized coordinates are required for a dense box head:
        # a purely translation-equivariant feature cannot regress global xyxy
        # values and was the actionable cause of repair1's near-zero IoU.
        self.objectness=nn.Conv2d(130,1,1)
        self.box_delta=nn.Conv2d(130,4,1)
        self.quality=nn.Conv2d(130,1,1)
        self.global_proj=nn.Sequential(nn.Linear(128,256),nn.LayerNorm(256),nn.GELU())
        self.raw_proj=nn.Sequential(nn.Linear(256,768),nn.LayerNorm(768))
        self.support_residual=nn.Sequential(nn.Linear(1536,512),nn.GELU(),nn.Linear(512,768))
        self.lifecycle=nn.Linear(256+1,3)       # birth/continue/terminate
        self.controller=nn.Sequential(nn.Linear(6,64),nn.GELU(),nn.Linear(64,3))

    def visual(self,x):
        f=self.backbone(x)
        g=F.adaptive_avg_pool2d(f,1).flatten(1)
        tr=F.normalize(self.global_proj(g),dim=-1)
        raw=F.normalize(self.raw_proj(self.global_proj(g)),dim=-1)
        return f,tr,raw

    def forward(self,x, age=None, support=None, support_valid=None, support_quality=None):
        f,tr,raw=self.visual(x)
        h,w=f.shape[-2:]
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=f.device,dtype=f.dtype),torch.linspace(-1,1,w,device=f.device,dtype=f.dtype),indexing='ij')
        hf=torch.cat([f,xx[None,None].expand(f.shape[0],-1,-1,-1),yy[None,None].expand(f.shape[0],-1,-1,-1)],1)
        out={"feature_map":f,"track_embedding":tr,"raw_state":raw,
             "objectness_logit":self.objectness(hf).squeeze(1),
             "bbox_logits":self.box_delta(hf),
             "quality_logit":self.quality(hf).squeeze(1)}
        if age is None: age=torch.zeros((x.shape[0],1),device=x.device)
        elif age.ndim==1: age=age[:,None]
        out["lifecycle_logits"]=self.lifecycle(torch.cat([self.global_proj(F.adaptive_avg_pool2d(f,1).flatten(1)),age],1))
        state=raw
        if support is not None:
            if support.ndim==1: support=support[None].expand_as(raw)
            if support.shape[0]!=raw.shape[0]: support=support.expand(raw.shape[0],-1)
            delta=0.10*torch.tanh(self.support_residual(torch.cat([raw,support],1)))
            candidate=F.normalize(raw+delta,dim=-1)
            if support_valid is None: support_valid=torch.ones((raw.shape[0],1),device=raw.device,dtype=torch.bool)
            if support_valid.ndim==1: support_valid=support_valid[:,None]
            state=torch.where(support_valid, candidate, raw)
        out["semantic_state"]=state
        if support_quality is None: support_quality=torch.zeros((raw.shape[0],1),device=raw.device)
        elif support_quality.ndim==1: support_quality=support_quality[:,None]
        evidence=(state*raw).sum(1,keepdim=True)
        persistence=torch.ones_like(evidence)*torch.clamp(age,0,1)
        uncertainty=1.0-evidence.clamp(-1,1)
        contradiction=(1.0-evidence).clamp(0,2)
        mot_safety=torch.sigmoid(out["quality_logit"].flatten(1).mean(1,keepdim=True))
        ctrl_in=torch.cat([evidence,persistence,uncertainty,contradiction,support_quality,mot_safety],1)
        out["controller_logits"]=self.controller(ctrl_in)
        out["controller_input"]=ctrl_in
        return out

    @staticmethod
    def decode_boxes(out, topk=10):
        """Decode class-agnostic boxes in normalized xyxy coordinates."""
        obj=torch.sigmoid(out["objectness_logit"])
        b=torch.sigmoid(out["bbox_logits"])
        n,_,h,w=b.shape
        flat=obj.flatten(1)
        k=min(topk,flat.shape[1]); vals,idx=torch.topk(flat,k,dim=1)
        yy=(idx//w).float(); xx=(idx%w).float()
        allb=[]
        for i in range(n):
            bi=b[i].reshape(4,-1)[:,idx[i]].transpose(0,1)
            allb.append(bi.clamp(0,1))
        return torch.stack(allb), vals
