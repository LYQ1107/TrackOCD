"""PHE-Track: trajectory-level adaptation of the official PHE (Prototypical Hash
Encoding) architecture. Keeps CPG (multi-prototype per class), DCE (hash encoding)
and the official losses; the input is a precomputed frozen track embedding."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class PrototypeMask(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, X):
        if self.training:
            mask = torch.bernoulli(torch.full_like(X, 1 - self.p))
            return X * mask
        return X * (1 - self.p)


class HASHHead(nn.Module):
    def __init__(self, in_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, code_dim=12):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            layers.append(nn.BatchNorm1d(bottleneck_dim))
            layers.append(nn.GELU())
            self.mlp = nn.Sequential(*layers)
        self.hash = nn.Linear(bottleneck_dim, code_dim, bias=False)
        self.bn_h = nn.BatchNorm1d(code_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.hash(self.mlp(x))


class TrackFeatureAdapter(nn.Module):
    """Small trainable adapter replacing the fine-tuned last ViT block of official PHE."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.net(x)


class PPNetTrack(nn.Module):
    def __init__(
        self,
        in_dim=768,
        prototype_dim=768,
        num_classes=78,
        global_proto_per_class=10,
        hash_code_length=12,
        mask_theta=0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.prototype_dim = prototype_dim
        self.num_classes = num_classes
        self.global_proto_per_class = global_proto_per_class
        self.num_prototypes_global = num_classes * global_proto_per_class
        self.epsilon = 1e-4
        self.prototype_activation_function = "log"
        self.hash_code_length = hash_code_length

        self.features = TrackFeatureAdapter(in_dim)
        self.add_on_layers = nn.Sequential(nn.Linear(in_dim, prototype_dim), nn.GELU())

        self.prototype_vectors_global = nn.Parameter(
            torch.rand(self.num_prototypes_global, prototype_dim), requires_grad=True
        )
        self.last_layer_global = nn.Linear(self.num_prototypes_global, num_classes, bias=False)
        self.last_layer_global.weight.requires_grad = False
        self.prototype_class_identity_global = torch.zeros(
            self.num_prototypes_global, num_classes
        )
        for j in range(self.num_prototypes_global):
            self.prototype_class_identity_global[j, j // global_proto_per_class] = 1

        self.hash_head = HASHHead(in_dim=prototype_dim, code_dim=hash_code_length)
        self.prototypemask = PrototypeMask(p=mask_theta)
        self._init_weights()
        self.set_last_layer_incorrect_connection(-0.5)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_last_layer_incorrect_connection(self, incorrect_strength=-0.5):
        pos = torch.t(self.prototype_class_identity_global)
        neg = 1 - pos
        self.last_layer_global.weight.data.copy_(
            1.0 * pos + incorrect_strength * neg
        )

    def get_activations(self, tokens, prototype_vectors):
        tokens = F.normalize(tokens, dim=-1)
        pv = F.normalize(prototype_vectors, dim=-1)
        diff = tokens.unsqueeze(1) - pv.unsqueeze(0)
        distances = diff.square().sum(dim=2).sqrt()
        return torch.log((distances + 1) / (distances + self.epsilon))

    def forward(self, x):
        cls_tokens = self.add_on_layers(self.features(x))
        hash_feat = self.hash_head(cls_tokens)
        if not self.training:
            return hash_feat
        global_activations = self.get_activations(cls_tokens, self.prototype_vectors_global)
        global_activations = self.prototypemask(global_activations)
        logits_global = self.last_layer_global(global_activations)
        return logits_global, hash_feat

    def prototype_class_centers(self):
        """Mean prototype vector per class, used to derive hash centers."""
        centers = []
        with torch.no_grad():
            for c in range(self.num_classes):
                sl = slice(c * self.global_proto_per_class, (c + 1) * self.global_proto_per_class)
                centers.append(self.prototype_vectors_global[sl].mean(0))
        return torch.stack(centers)

    def hash_centers(self):
        centers = self.prototype_class_centers()
        return torch.tanh(self.hash_head(centers) * 3).sign()


def cos_eps_loss(u, y, hash_center):
    u_norm = F.normalize(u)
    centers_norm = F.normalize(hash_center)
    cos_sim = torch.matmul(u_norm, centers_norm.t())
    return F.cross_entropy(cos_sim, y)


def sep_loss(prototype_centers, L=12, dis_max=3, alpha=0.95):
    labels = torch.arange(prototype_centers.shape[0], device=prototype_centers.device)
    dot = torch.matmul(prototype_centers, prototype_centers.t())
    hamming = 0.5 * (L * alpha - dot)
    mask = labels.unsqueeze(1) != labels.unsqueeze(0)
    return (F.relu(dis_max - hamming) * mask.float()).sum(-1).mean()


def binomial_coefficient(n, k):
    return math.comb(n, k)


def get_dis_max(L, y_u):
    target = (2**L) / y_u if y_u > 0 else 2**L
    for d in range(2, L + 1):
        lower = binomial_coefficient(L, d - 2)
        upper = binomial_coefficient(L, d - 1)
        if lower <= target <= upper:
            return d
    return 0
