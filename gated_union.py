"""Reliability gates for multi-hint fusion: each hint is weighted per pixel by how much it should be trusted there.

- Box and scribble (SpatialHintGate): a per-pixel gate on each hint's term in the ControlNet-Union, just before the
  terms are summed into the sample. ControlNet++ weights a hint with one per-channel offset for the whole image; the
  gate makes the weight per pixel. It sees the hint's features, the sample (noisy latent and visible-region
  condition) and the timestep, and is multiplied by the hint's validity: outside it, a missing line means
  "unknown", not "no edge".
- Reference (GatedReferenceAttnProcessor): in every UNet self-attention layer, a per-token gate between attending
  with and without the reference tokens: out = alone + gate * (with_reference - alone).

Both gates start almost open (sigmoid(4) = 0.98), so the untrained gated model is close to the ungated one.

Reliability control at inference (set_reliability): beta is added to every gate logit, so beta > 0 follows the hints
more and beta < 0 less; gates_off sets every gate to 1 (the validity map still applies), so every hint is trusted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from controlnet_union import ControlNetUnionModel
from trainer_reference_only import ReferenceAttnControl, ReferenceAttnProcessor

GATE_INIT_BIAS = 4.0


class SpatialHintGate(nn.Module):
    """Three 3x3 convolutions (dilations 1, 2, 4: a 15x15 latent field, 120 px) over [hint term, sample], with the
    timestep added after the first."""

    def __init__(self, channels, time_embed_dim, hidden=64):
        super().__init__()
        self.conv_in = nn.Conv2d(2 * channels, hidden, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, hidden)
        self.conv_mid = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2)
        self.conv_out = nn.Conv2d(hidden, 1, 3, padding=4, dilation=4)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.constant_(self.conv_out.bias, GATE_INIT_BIAS)

    def forward(self, term, sample, emb, validity, beta=0.0, off=False):
        if off:
            return validity
        h = self.conv_in(torch.cat([term, sample], dim=1)) + self.time_proj(F.silu(emb))[:, :, None, None]
        h = self.conv_mid(F.silu(h))
        return torch.sigmoid(self.conv_out(F.silu(h)) + beta) * validity


class GatedControlNetUnionModel(ControlNetUnionModel):
    """ControlNet-Union with a SpatialHintGate per control type. The last gates are kept in self.gates."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        time_embed_dim = self.config.block_out_channels[0] * 4
        self.hint_gates = nn.ModuleList(
            [SpatialHintGate(self.config.block_out_channels[0], time_embed_dim) for _ in range(self.config.num_control_type)]
        )
        self.gates = None
        self.gate_beta = 0.0  # reliability control, see set_reliability
        self.gates_off = False

    def hint_terms(self, sample, emb, controlnet_cond, control_type_idx, conditioning_scale, validity=None, **kwargs):
        """validity: (B, 1, H, W) at the hint resolution, 1 where the hints are known. None means everywhere."""
        terms = super().hint_terms(sample, emb, controlnet_cond, control_type_idx, conditioning_scale)
        if validity is None:
            validity = torch.ones_like(sample[:, :1])
        else:
            validity = F.interpolate(validity.to(sample.dtype), size=sample.shape[-2:], mode='area')

        self.gates = [self.hint_gates[idx](term, sample, emb, validity, self.gate_beta, self.gates_off) for term, idx in zip(terms, control_type_idx)]
        return [gate * term for gate, term in zip(self.gates, terms)]


class GatedReferenceAttnControl(ReferenceAttnControl):
    def __init__(self):
        super().__init__()
        self.scale = 1.0  # multiplies the reference gate at inference
        self.gates = None  # a list to collect the reference gates of every layer in a read pass
        self.beta = 0.0  # reliability control, see set_reliability
        self.gates_off = False


class GatedReferenceAttnProcessor(ReferenceAttnProcessor):
    """In 'read' mode, mixes self-attention with and without the reference tokens by attn.reference_gate."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        control = self.control
        if control.mode != 'read':
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        bank = self.bank.repeat(hidden_states.shape[0] // self.bank.shape[0], 1, 1)
        with_reference = self.processor(attn, hidden_states, encoder_hidden_states=torch.cat([hidden_states, bank], dim=1))
        alone = self.processor(attn, hidden_states)

        if control.gates_off:
            gate = torch.ones_like(hidden_states[..., :1]) * control.scale
        else:
            gate = torch.sigmoid(attn.reference_gate(hidden_states) + control.beta) * control.scale
        if control.drop is not None:
            gate = gate * (~control.drop)[:, None, None]
        if control.gates is not None:
            control.gates.append(gate)

        return alone + gate * (with_reference - alone)


def add_reference_gates(unet):
    """Adds a reference_gate (Linear(dim, 1)) to every self-attention layer. Returns them."""
    gates = nn.ModuleList()
    for name, module in unet.named_modules():
        if name.endswith('attn1'):
            module.reference_gate = nn.Linear(module.to_q.base.in_features, 1)
            nn.init.zeros_(module.reference_gate.weight)
            nn.init.constant_(module.reference_gate.bias, GATE_INIT_BIAS)
            gates.append(module.reference_gate)
    return gates


def set_reliability(controlnet, control, beta=0.0, gates_off=False):
    """Reliability control at inference. beta: offset on every gate logit (spatial and reference); positive follows the
    hints more closely, negative falls back on the visible part. gates_off: every gate is 1, so every hint is trusted
    (spatial hints still only where their validity map is 1). set_reliability(controlnet, control) restores the default."""
    controlnet.gate_beta, controlnet.gates_off = float(beta), bool(gates_off)
    control.beta, control.gates_off = float(beta), bool(gates_off)
