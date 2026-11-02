"""Shared Transformer recurrence and explicit parameter-occurrence estimators.

Architecture equations: arXiv2609.39967v1. HRM shape conventions were inspected
at ac15626f8db096a63c775b84c9dc868776a6feda (Apache2.0; see
third_party/HRM_APACHE_LICENSE); no upstream model
module is imported. Weighted norms, four-class head, fixed ACT and Torch math
attention are explicit implementation choices, not verified recipe code.
"""
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.func import functional_call

EVENTS = tuple(e for h in range(1, 5) for e in (f'L{h}.1', f'L{h}.2', f'H{h}'))
EARLY_LATE = frozenset(EVENTS[:3] + EVENTS[-3:])


@dataclass(frozen=True)
class PilotConfig:
    vocab_size: int = 13
    pad_id: int = 0
    seq_len: int = 19
    hidden_size: int = 512
    layers: int = 4
    heads: int = 8
    expansion: float = 4.0
    ff_multiple: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    update_bound: float = 0.7
    bound_eps: float = 1e-8
    dropout_core: float = .025
    dropout_state: float = .010
    noise_scale: float = .005
    forward_dtype: str = 'bfloat16'
    engineering_fixture: bool = False

    def __post_init__(self):
        if self.vocab_size < 2 or not 0 <= self.pad_id < self.vocab_size:
            raise ValueError('invalid token vocabulary')
        if self.seq_len < 1 or self.layers < 1 or self.heads < 1 or self.hidden_size % self.heads or (self.hidden_size // self.heads) % 2:
            raise ValueError('positive dimensions and even RoPE head width required')
        if self.ff_multiple < 1 or self.expansion <= 0 or self.norm_eps <= 0 or self.bound_eps <= 0 or self.update_bound <= 0 or self.noise_scale < 0:
            raise ValueError('invalid architecture constants')
        if not all(0 <= p < 1 for p in (self.dropout_core, self.dropout_state)):
            raise ValueError('dropout must be in [0,1)')
        if self.forward_dtype not in ('float32', 'bfloat16', 'float64') or (self.forward_dtype == 'float64' and not self.engineering_fixture):
            raise ValueError('FP64 is an engineering fixture only')

    @property
    def ff_width(self):
        return math.ceil(round(self.expansion * self.hidden_size * 2 / 3) / self.ff_multiple) * self.ff_multiple


@dataclass(frozen=True)
class BackwardPolicy:
    kind: str
    kl: int = 2
    kh: int = 4
    mask: frozenset = frozenset()

    def __post_init__(self):
        if self.kind == 'suffix':
            if self.kl not in (1, 2) or self.kh not in (1, 2, 3, 4):
                raise ValueError('suffix axes are KL=1..2, KH=1..4')
        elif self.kind != 'occurrence' or not self.mask or not self.mask.issubset(EVENTS):
            raise ValueError('invalid occurrence policy')

    @classmethod
    def for_arm(cls, arm):
        if arm == 'S22': return cls('suffix', kl=2, kh=2)
        if arm == 'S13': return cls('suffix', kl=1, kh=3)
        if arm == 'Oearlylate': return cls('occurrence', mask=EARLY_LATE)
        if arm == 'FULL': return cls('suffix', kl=2, kh=4)
        if arm == 'ALL': return cls('occurrence', mask=frozenset(EVENTS))
        raise ValueError('unknown arm')


def _draw(shape, seed, trajectory, name, *, device, dtype, normal=False):
    key = bytes.fromhex('67726164706174682d70696c6f742d7631') + f':{seed}:{trajectory}:{name}'.encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)
    generator = torch.Generator(device=device).manual_seed(value)
    draw = torch.randn if normal else torch.rand
    return draw(shape, generator=generator, device=device, dtype=dtype)


def stochastic_plan(config, shape, seed, trajectory, *, device, dtype, training):
    """One dropout draw per trajectory, reused at every call and all8 ACT steps."""
    batch, seq = shape
    masks = {}
    for name, width in [('input', config.hidden_size), ('low', config.hidden_size), ('high', config.hidden_size)] + [(f'{i}.{part}', width) for i in range(config.layers) for part, width in [('qkv', 3*config.hidden_size), ('attention', config.hidden_size), ('intermediate', config.ff_width), ('feedforward', config.hidden_size)]]:
        p = config.dropout_state if name in ('low', 'high') else config.dropout_core
        mask_shape = (batch, 1 if name in ('low', 'high') else seq, width)
        masks[name] = ((_draw(mask_shape, seed, trajectory, name, device=device, dtype=dtype) >= p).to(dtype) / (1-p)) if training and p else torch.ones(mask_shape, device=device, dtype=dtype)
    return masks


class CastLinear(nn.Linear):
    def forward(self, value):
        return F.linear(value, self.weight.to(value.dtype), None if self.bias is None else self.bias.to(value.dtype))


class WeightedRMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value):
        work = value if value.dtype == torch.float64 else value.float()
        normed = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)
        return normed.to(value.dtype) * self.weight.to(value.dtype)


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.qkv = CastLinear(width, 3*width, bias=False)
        self.attention_out = CastLinear(width, width, bias=False)
        self.gate_up = CastLinear(width, 2*config.ff_width, bias=False)
        self.down = CastLinear(config.ff_width, width, bias=False)
        self.attention_norm = WeightedRMSNorm(width, config.norm_eps)
        self.feedforward_norm = WeightedRMSNorm(width, config.norm_eps)
        head_width = width // config.heads
        frequencies = torch.outer(torch.arange(config.seq_len, dtype=torch.float64), config.rope_theta ** (-torch.arange(0, head_width, 2, dtype=torch.float64) / head_width))
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer('rope_cos', angles.cos(), persistent=False)
        self.register_buffer('rope_sin', angles.sin(), persistent=False)

    def forward(self, value, masks, index, valid):
        batch, seq, width = value.shape
        qkv = (self.qkv(value) * masks[f'{index}.qkv']).reshape(batch, seq, 3, self.config.heads, width // self.config.heads)
        query, key, val = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        cos = self.rope_cos[:seq].to(value.dtype)[None, None]
        sin = self.rope_sin[:seq].to(value.dtype)[None, None]
        def rotate(tensor):
            first, second = tensor.chunk(2, dim=-1)
            return tensor*cos + torch.cat((-second, first), dim=-1)*sin
        query, key = rotate(query), rotate(key)
        work_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        scores = (query.to(work_dtype) @ key.to(work_dtype).transpose(-1, -2)) / math.sqrt(width // self.config.heads)
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        attended = (scores.softmax(-1).to(value.dtype) @ val).transpose(1, 2).reshape(batch, seq, width)
        value = self.attention_norm(value + self.attention_out(attended) * masks[f'{index}.attention'])
        gate, up = self.gate_up(value).chunk(2, dim=-1)
        intermediate = F.silu(gate) * up * masks[f'{index}.intermediate']
        return self.feedforward_norm(value + self.down(intermediate) * masks[f'{index}.feedforward'])


class RecurrentCore(nn.Module):
    """Masked partition: shared stack, low gate, low/high state norms."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.layers)])
        self.low_gate = CastLinear(config.hidden_size, 1, bias=True)
        self.low_norm = WeightedRMSNorm(config.hidden_size, config.norm_eps)
        self.high_norm = WeightedRMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, low, high, inputs, is_low, masks, valid):
        value = low + high + (inputs if is_low else 0)
        candidate = value
        for index, layer in enumerate(self.layers):
            candidate = layer(candidate, masks, index, valid)
        if is_low:
            delta = candidate - low
            work_delta = delta if delta.dtype == torch.float64 else delta.float()
            work_low = low if low.dtype == torch.float64 else low.float()
            ratio = work_delta.norm(dim=-1, keepdim=True) / (work_low.norm(dim=-1, keepdim=True) + self.config.bound_eps)
            divisor = (ratio / self.config.update_bound).clamp_min(1).detach().to(delta.dtype)
            output = self.low_norm(low + torch.sigmoid(self.low_gate(value)) * delta / divisor)
            return output * masks['low']
        return self.high_norm(candidate) * masks['high']


class ArithmeticModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.initial_low = nn.Parameter(torch.empty(config.hidden_size))
        self.initial_high = nn.Parameter(torch.empty(config.hidden_size))
        self.core = RecurrentCore(config)
        self.operator_head = CastLinear(config.hidden_size, 4, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 1 / math.sqrt(module.in_features)
                nn.init.trunc_normal_(module.weight, std=std, a=-2*std, b=2*std)
                if module.bias is not None: nn.init.zeros_(module.bias)
        std = 1 / math.sqrt(config.hidden_size)
        nn.init.trunc_normal_(self.embedding.weight, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.initial_low, std=1, a=-2, b=2)
        nn.init.trunc_normal_(self.initial_high, std=1, a=-2, b=2)
        with torch.no_grad(): self.core.low_gate.weight.mul_(.1)

    def prepare(self, token_ids, carry, masks, seed, trajectory, act_step, training):
        if token_ids.ndim != 2 or token_ids.shape[1] > self.config.seq_len or token_ids.dtype != torch.long or token_ids.min() < 0 or token_ids.max() >= self.config.vocab_size:
            raise ValueError('invalid encoded token batch')
        valid = token_ids != self.config.pad_id
        if not valid.any(-1).all(): raise ValueError('empty input sequence')
        dtype = getattr(torch, self.config.forward_dtype)
        inputs = self.embedding(token_ids).to(dtype) * math.sqrt(self.config.hidden_size) * masks['input']
        if carry is None:
            low = self.initial_low.to(dtype).expand_as(inputs)
            high = self.initial_high.to(dtype).expand_as(inputs)
        else:
            low, high = (state.detach() for state in carry)
            if low.shape != inputs.shape or high.shape != inputs.shape: raise ValueError('carry shape mismatch')
        if training and self.config.noise_scale:
            values = []
            for name, state in [('input', inputs), ('low', low), ('high', high)]:
                norm_state = state if dtype == torch.float64 else state.float()
                scale = norm_state.norm(dim=-1, keepdim=True).detach().to(dtype)
                noise = _draw(state.shape, seed, trajectory, f'noise:{act_step}:{name}', device=state.device, dtype=dtype, normal=True)
                values.append(state + self.config.noise_scale * scale * noise)
            inputs, low, high = values
        return inputs, low, high, valid

    def forward_act(self, token_ids, policy, *, carry=None, masks=None, seed=0, trajectory=0, act_step=0, trace=None, trace_values=True):
        if masks is None:
            masks = stochastic_plan(self.config, token_ids.shape, seed, trajectory, device=token_ids.device, dtype=getattr(torch, self.config.forward_dtype), training=self.training)
        inputs, low, high, valid = self.prepare(token_ids, carry, masks, seed, trajectory, act_step, self.training)
        parameter_views = {name: value.detach() for name, value in self.core.named_parameters()}
        for h in range(1, 5):
            if policy.kind == 'suffix' and policy.kh < 4 and h == 5-policy.kh:
                low, high = low.detach(), high.detach()
            for stage in (1, 2, 3):
                label = f'L{h}.{stage}' if stage < 3 else f'H{h}'
                recorded = policy.kind == 'occurrence' or (h > 4-policy.kh and (stage == 3 or stage > 2-policy.kl))
                selected = recorded if policy.kind == 'suffix' else label in policy.mask
                if not recorded:
                    with torch.no_grad(): output = self.core(low, high, inputs, stage < 3, masks, valid)
                elif selected:
                    output = self.core(low, high, inputs, stage < 3, masks, valid)
                else:
                    output = functional_call(self.core, parameter_views, (low, high, inputs, stage < 3, masks, valid), strict=False)
                if trace is not None:
                    with torch.no_grad():
                        previous = low if stage < 3 else high
                        work_output = output.detach() if output.dtype == torch.float64 else output.detach().float()
                        work_previous = previous.detach().to(work_output.dtype)
                        displacement = ((work_output-work_previous).norm(dim=-1) / (work_previous.norm(dim=-1)+1e-8)).mean().item()
                    item = {'event': label, 'recorded': recorded, 'selected_core': selected, 'relative_displacement_mean': displacement}
                    if trace_values: item['value'] = output.detach().clone()
                    trace.append(item)
                if stage < 3: low = output
                else: high = output
        return self.operator_head(high), (low.detach(), high.detach())
