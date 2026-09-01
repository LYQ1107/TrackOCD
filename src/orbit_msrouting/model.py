"""ORBIT-MSRouting model: state-conditioned known gate (Phase 4G).

G0: Phase 4F M2 gate (evidence only).
G1: evidence + selected memory-state features concatenated into the gate.
G2: evidence gate with a small state-adaptive residual calibration
    gate_logit_corrected = gate_logit - b(S_t), where b is a small MLP.

The aggregator and compatibility head are identical to ORBIT-MDC.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.orbit_fc.known_gate import KnownGate
from src.orbit_mdc.model import ORBITMDCModel


class StateCalibration(nn.Module):
    """Small MLP mapping memory-state features to a scalar bias b(S_t)."""

    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ORBITMSRoutingModel(ORBITMDCModel):
    def __init__(self, dim=768, bottleneck=128, gate_dim=11, reuse_dim=11,
                 hidden=64, use_adapter=True, compat_dim=7, birth_dim=0,
                 gate_mode="G0", state_dim=0):
        super().__init__(dim, bottleneck, gate_dim, reuse_dim, hidden,
                         use_adapter, compat_dim, birth_dim)
        assert gate_mode in ("G0", "G1", "G2")
        self.gate_mode = gate_mode
        self.state_dim = state_dim
        if gate_mode == "G1":
            self.gate = KnownGate(gate_dim + state_dim, hidden)
        if gate_mode == "G2":
            self.calib = StateCalibration(state_dim, hidden=16)

    def gate_logit(self, evidence, state_feats=None):
        """Return the gate logit for the current decision.

        G0: gate(evidence)
        G1: gate([evidence; state_feats])
        G2: gate(evidence) - calib(state_feats)
        """
        if self.gate_mode == "G0":
            return self.gate_forward(evidence)
        if self.gate_mode == "G1":
            if state_feats is None:
                raise RuntimeError("G1 requires state features")
            x = torch.cat([evidence, state_feats], dim=1)
            return self.gate_forward(x)
        if self.gate_mode == "G2":
            if state_feats is None:
                raise RuntimeError("G2 requires state features")
            return self.gate_forward(evidence) - self.calib(state_feats)
        raise RuntimeError(f"unknown gate_mode {self.gate_mode}")


def load_msrouting_model(path, device="cpu", gate_mode=None, state_dim=None,
                         state_feats=None):
    ck = torch.load(path, map_location="cpu")
    ck_gm = ck.get("gate_mode", "G0")
    if gate_mode is not None:
        ck_gm = gate_mode
    if state_dim is None:
        state_dim = ck.get("state_dim", 0)
    if state_dim == 0 and state_feats is not None:
        state_dim = len(state_feats)
    # gate_dim is always the evidence dimension (11); G1's constructor adds
    # the state dimension itself.
    gate_dim = ck.get("gate_dim", 11)
    model = ORBITMSRoutingModel(
        dim=768, bottleneck=ck.get("bottleneck", 128),
        gate_dim=gate_dim,
        # ORBIT-MDC M1/M2 checkpoints were trained with reuse_dim=13
        # (mem_scale_norm), even though the metadata does not record it.
        reuse_dim=ck.get("reuse_dim", 13),
        hidden=64, use_adapter=True,
        compat_dim=ck.get("compat_dim", 7),
        birth_dim=ck.get("birth_dim", 0),
        gate_mode=ck_gm, state_dim=state_dim)
    sd = model.state_dict()
    # G1: copy the base gate weights into the first 11 columns and zero the
    # state-feature columns so the model starts as the M2 gate.
    if ck_gm == "G1" and state_dim > 0:
        base_w = ck["state_dict"]["gate.net.0.weight"]
        base_b = ck["state_dict"]["gate.net.0.bias"]
        with torch.no_grad():
            model.gate.net[0].weight.zero_()
            model.gate.net[0].bias.zero_()
            model.gate.net[0].weight[:, :base_w.shape[1]] = base_w
            model.gate.net[0].bias.copy_(base_b)
            for src_key, dst_key in [
                    ("gate.net.1.weight", "gate.net.1.weight"),
                    ("gate.net.1.bias", "gate.net.1.bias"),
                    ("gate.net.2.weight", "gate.net.2.weight"),
                    ("gate.net.2.bias", "gate.net.2.bias"),
                    ("gate.net.3.weight", "gate.net.3.weight"),
                    ("gate.net.3.bias", "gate.net.3.bias"),
                    ("gate.net.4.weight", "gate.net.4.weight"),
                    ("gate.net.4.bias", "gate.net.4.bias")]:
                if src_key in ck["state_dict"] and dst_key in sd:
                    model.state_dict()[dst_key].copy_(
                        ck["state_dict"][src_key])
    if ck_gm == "G2" and state_dim > 0:
        # start as the M2 gate: zero calibration output
        with torch.no_grad():
            model.calib.net[-1].weight.zero_()
            model.calib.net[-1].bias.zero_()
    sd = model.state_dict()
    for k, v in ck["state_dict"].items():
        if k in sd and v.shape == sd[k].shape:
            sd[k] = v
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, ck


def build_msrouting_model(init_checkpoint, gate_mode, state_names, device="cpu"):
    """Build a trainable ORBIT-MSRouting model initialized from an ORBIT-MDC
    checkpoint (G0: exact copy; G1/G2: base gate copied, state paths zeroed)."""
    ck = torch.load(init_checkpoint, map_location="cpu")
    state_dim = len(state_names) if gate_mode in ("G1", "G2") else 0
    model = ORBITMSRoutingModel(
        dim=768, bottleneck=ck.get("bottleneck", 128),
        gate_dim=11, reuse_dim=ck.get("reuse_dim", 13), hidden=64,
        use_adapter=True, compat_dim=ck.get("compat_dim", 7),
        birth_dim=ck.get("birth_dim", 0), gate_mode=gate_mode,
        state_dim=state_dim).to(device)
    if gate_mode == "G1" and state_dim > 0:
        with torch.no_grad():
            base_w = ck["state_dict"]["gate.net.0.weight"]
            base_b = ck["state_dict"]["gate.net.0.bias"]
            model.gate.net[0].weight.zero_()
            model.gate.net[0].bias.zero_()
            model.gate.net[0].weight[:, :base_w.shape[1]] = base_w
            model.gate.net[0].bias.copy_(base_b)
    if gate_mode == "G2" and state_dim > 0:
        with torch.no_grad():
            model.calib.net[-1].weight.zero_()
            model.calib.net[-1].bias.zero_()
    sd = model.state_dict()
    for k, v in ck["state_dict"].items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
    model.load_state_dict(sd)
    return model, ck
