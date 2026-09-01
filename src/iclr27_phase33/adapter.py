import torch
import torch.nn.functional as F
from torch import nn
class QueryConditionedAdapter(nn.Module):
 def __init__(self,dim=768,hidden=256):
  super().__init__(); self.delta=nn.Sequential(nn.LayerNorm(dim*2),nn.Linear(dim*2,hidden),nn.GELU(),nn.Linear(hidden,dim)); self.alpha=nn.Parameter(torch.tensor(-6.0))
 def forward(self,raw,support=None):
  if support is None or support.numel()==0: return F.normalize(raw,dim=-1)
  s=support.mean(dim=-2) if support.ndim==3 else support; d=self.delta(torch.cat([raw,s],dim=-1)); a=0.2*torch.sigmoid(self.alpha); return F.normalize(raw+a*torch.tanh(d),dim=-1)
