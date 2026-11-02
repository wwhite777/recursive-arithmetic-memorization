"""Read-only project primitives and the experiment single-pass/ACT8 accumulation adapters."""
from dataclasses import dataclass
from functools import partial
import importlib.util
import math
import sys
from types import SimpleNamespace

from .common import PROJECT, require
sys.path.insert(0, str(PROJECT/'src'))
import torch
from reasoning.model import ArithmeticModel, PilotConfig, BackwardPolicy, stochastic_plan
from reasoning.model import _draw, CastLinear, TransformerLayer
from reasoning.trainer import PilotTrainer, supervised_loss, batch_identity


@dataclass(frozen=True)
class TrainingConfig:
    total_updates: int
    physical_batch: int = 1024
    act_steps: int = 8
    peak_lr: float = 5e-4
    warmup_updates: int = 2000
    min_lr_ratio: float = .01
    beta1: float = .9
    beta2: float = .95
    weight_decay: float = 1.0
    clip_norm: float = 1.0
    ema_decay: float = .999
    seed: int = 101
    engineering_fixture: bool = False
    effective_batch: int = None

    def __post_init__(self):
        require(self.total_updates > 0 and self.physical_batch > 0 and self.act_steps in (8, 16), 'training dimensions')
        if self.effective_batch is not None:
            require(type(self.effective_batch) is int and self.effective_batch > 0
                    and self.effective_batch % self.physical_batch == 0
                    and self.effective_batch // self.physical_batch in (1, 2, 4), 'C2 accumulation K=1,2,4')
            if not self.engineering_fixture:
                require(self.effective_batch == 4096, 'C2 effective batch4096')
                require(self.total_updates % self.act_steps == 0, 'complete trajectories')
        elif not self.engineering_fixture:
            require((self.physical_batch, self.act_steps) in ((1024, 8), (4096, 16)), 'fixed candidate schedules')
            require(self.total_updates % self.act_steps == 0, 'complete trajectories')
        require(type(self.seed) is int and -(2**63) <= self.seed < 2**64, 'integer seed required')


class NonrecursiveModel(torch.nn.Module):
    """Embedding -> one shared four-layer stack -> head; no recurrent parameters."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.core = torch.nn.Module()
        self.core.layers = torch.nn.ModuleList([TransformerLayer(config) for _ in range(config.layers)])
        self.operator_head = CastLinear(config.hidden_size, 4, bias=False)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                std = 1 / math.sqrt(module.in_features)
                torch.nn.init.trunc_normal_(module.weight, std=std, a=-2*std, b=2*std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        std = 1 / math.sqrt(config.hidden_size)
        torch.nn.init.trunc_normal_(self.embedding.weight, std=std, a=-2*std, b=2*std)

    def forward(self, token_ids, *, masks=None, seed=0, trajectory=0, act_step=0):
        if token_ids.ndim != 2 or token_ids.shape[1] > self.config.seq_len or token_ids.dtype != torch.long or token_ids.min() < 0 or token_ids.max() >= self.config.vocab_size:
            raise ValueError('invalid encoded token batch')
        valid = token_ids != self.config.pad_id
        require(valid.any(-1).all(), 'empty input sequence')
        dtype = getattr(torch, self.config.forward_dtype)
        if masks is None:
            masks = stochastic_plan(self.config, token_ids.shape, seed, trajectory,
                                    device=token_ids.device, dtype=dtype, training=self.training)
        value = self.embedding(token_ids).to(dtype) * math.sqrt(self.config.hidden_size) * masks['input']
        if self.training and self.config.noise_scale:
            work = value if dtype == torch.float64 else value.float()
            scale = work.norm(dim=-1, keepdim=True).detach().to(dtype)
            value = value + self.config.noise_scale * scale * _draw(value.shape, seed, trajectory,
                f'noise:{act_step}:input', device=value.device, dtype=dtype, normal=True)
        for index, layer in enumerate(self.core.layers):
            value = layer(value, masks, index, valid)
        return self.operator_head(value)


class AccumulationModel(ArithmeticModel):
    """The original forward map with slices of full-batch keyed noise draws."""
    step_noise = None

    def prepare(self, token_ids, carry, masks, seed, trajectory, act_step, training):
        if self.step_noise is None:
            return super().prepare(token_ids, carry, masks, seed, trajectory, act_step, training)
        inputs, low, high, valid = super().prepare(token_ids, carry, masks, seed, trajectory, act_step, False)
        if training and self.config.noise_scale:
            values = []
            for name, state in [('input', inputs), ('low', low), ('high', high)]:
                work = state if state.dtype == torch.float64 else state.float()
                scale = work.norm(dim=-1, keepdim=True).detach().to(state.dtype)
                values.append(state + self.config.noise_scale * scale * self.step_noise[name])
            inputs, low, high = values
        return inputs, low, high, valid


class Trainer(PilotTrainer):
    def step(self, batch, *, collect_displacement=False):
        if isinstance(self.model, NonrecursiveModel):
            return self.nonrecursive_step(batch)
        if self.config.effective_batch is not None:
            return self.accumulated_step(batch, collect_displacement=collect_displacement)
        if self.config.act_steps == 8:
            return super().step(batch, collect_displacement=collect_displacement)
        # Exact experiment step operations with only the held-batch ACT clock extended.
        require(self.updates < self.config.total_updates, 'update budget reached')
        require(batch['token_ids'].shape[0] == self.config.physical_batch, 'physical batch mismatch')
        identity = batch_identity(batch)
        act_step, trajectory = self.updates % 16, self.updates // 16
        if act_step == 0:
            self.carry = None
            self.current_batch_identity = identity
            self.masks = stochastic_plan(self.model.config, batch['token_ids'].shape, self.config.seed, trajectory,
                                         device=batch['token_ids'].device, dtype=getattr(torch, self.model.config.forward_dtype), training=True)
        else:
            require(identity == self.current_batch_identity, 'batch must remain identical for all16 updates')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        trace = [] if collect_displacement else None
        logits, carry = self.model.forward_act(batch['token_ids'], self.policy, carry=self.carry, masks=self.masks,
                                               seed=self.config.seed, trajectory=trajectory, act_step=act_step, trace=trace, trace_values=False)
        loss, _, correct = supervised_loss(logits, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite supervised loss')
        loss.backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm, error_if_nonfinite=True)
        lr = self.learning_rate(self.updates+1)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        self.optimizer_steps += 1
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                self.ema[name].lerp_(parameter, 1-self.config.ema_decay)
        self.ema_steps += 1
        self.carry = carry
        self.updates += 1
        return {'update': self.updates, 'act_step': act_step+1, 'trajectory': trajectory, 'batch_identity': identity,
                'loss': loss.item(), 'preclip_norm': raw_norm.item(), 'lr': lr, 'strict_operator_correct': int(correct.sum()),
                'total': len(correct), 'optimizer_steps': self.optimizer_steps, 'ema_steps': self.ema_steps, 'latent_displacement': trace}

    def nonrecursive_step(self, batch):
        require(self.config.effective_batch is None and self.config.act_steps == 8, 'NR requires standard held8 training')
        require(self.updates < self.config.total_updates, 'update budget reached')
        require(batch['token_ids'].shape[0] == self.config.physical_batch, 'physical batch mismatch')
        identity = batch_identity(batch)
        act_step, trajectory = self.updates % 8, self.updates // 8
        if act_step == 0:
            self.current_batch_identity = identity
            plan = stochastic_plan(self.model.config, batch['token_ids'].shape, self.config.seed, trajectory,
                device=batch['token_ids'].device, dtype=getattr(torch, self.model.config.forward_dtype), training=True)
            self.masks = {k: v for k, v in plan.items() if k not in ('low', 'high')}
        else:
            require(identity == self.current_batch_identity, 'batch must remain identical for all8 updates')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch['token_ids'], masks=self.masks, seed=self.config.seed,
                            trajectory=trajectory, act_step=act_step)
        loss, _, correct = supervised_loss(logits, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite supervised loss')
        loss.backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm, error_if_nonfinite=True)
        lr = self.learning_rate(self.updates+1)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        self.optimizer_steps += 1
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                self.ema[name].lerp_(parameter, 1-self.config.ema_decay)
        self.ema_steps += 1
        self.updates += 1
        return {'update': self.updates, 'act_step': act_step+1, 'trajectory': trajectory, 'batch_identity': identity,
                'loss': loss.item(), 'preclip_norm': raw_norm.item(), 'lr': lr, 'strict_operator_correct': int(correct.sum()),
                'total': len(correct), 'optimizer_steps': self.optimizer_steps, 'ema_steps': self.ema_steps, 'latent_displacement': None}

    def accumulated_step(self, batch, *, collect_displacement=False):
        require(self.updates < self.config.total_updates, 'update budget reached')
        size = self.config.effective_batch
        require(batch['token_ids'].shape[0] == size, 'effective batch mismatch')
        require(isinstance(self.model, AccumulationModel), 'C2 requires full-batch noise adapter')
        identity = batch_identity(batch)
        steps = self.config.act_steps
        act_step, trajectory = self.updates % steps, self.updates // steps
        if act_step == 0:
            self.carry = None
            self.current_batch_identity = identity
            self.masks = stochastic_plan(self.model.config, batch['token_ids'].shape, self.config.seed, trajectory,
                                         device=batch['token_ids'].device, dtype=getattr(torch, self.model.config.forward_dtype), training=True)
        else:
            require(identity == self.current_batch_identity, f'batch must remain identical for all{steps} updates')
        # Draw with the physical reference's full shape and keys. In particular,
        # randn on a smaller shape (or another device) is not an equivalent stream.
        noise = {}
        if self.model.config.noise_scale:
            shape = (*batch['token_ids'].shape, self.model.config.hidden_size)
            noise = {name: _draw(shape, self.config.seed, trajectory, f'noise:{act_step}:{name}',
                                device=batch['token_ids'].device, dtype=getattr(torch, self.model.config.forward_dtype), normal=True)
                     for name in ('input', 'low', 'high')}
        denominator = batch['target_mask'].sum()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        carries, loss_value, correct_count = [], 0., 0
        trace = [] if collect_displacement else None
        try:
            for start in range(0, size, self.config.physical_batch):
                end = start+self.config.physical_batch
                micro = {k: v[start:end] for k, v in batch.items()}
                masks = {k: v[start:end] for k, v in self.masks.items()}
                carry = None if self.carry is None else tuple(v[start:end] for v in self.carry)
                self.model.step_noise = {k: v[start:end] for k, v in noise.items()}
                micro_trace = [] if collect_displacement else None
                logits, next_carry = self.model.forward_act(micro['token_ids'], self.policy, carry=carry, masks=masks,
                    seed=self.config.seed, trajectory=trajectory, act_step=act_step, trace=micro_trace, trace_values=False)
                loss, _, correct = supervised_loss(logits, micro)
                # experiment token weighting, using the effective batch denominator.
                loss = loss * (micro['target_mask'].sum().to(loss.dtype) / denominator)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite supervised loss')
                loss.backward()
                loss_value += loss.item()
                correct_count += int(correct.sum())
                carries.append(next_carry)
                if micro_trace is not None:
                    if not trace:
                        trace.extend(dict(x, relative_displacement_mean=0.) for x in micro_trace)
                    for target, item in zip(trace, micro_trace):
                        target['relative_displacement_mean'] += item['relative_displacement_mean'] * (end-start)/size
        finally:
            self.model.step_noise = None
        # Exactly one clipping operation, optimizer step, EMA step and clock tick.
        raw_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm, error_if_nonfinite=True)
        lr = self.learning_rate(self.updates+1)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        self.optimizer_steps += 1
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                self.ema[name].lerp_(parameter, 1-self.config.ema_decay)
        self.ema_steps += 1
        self.carry = tuple(torch.cat([c[i] for c in carries], dim=0) for i in (0, 1))
        self.updates += 1
        return {'update': self.updates, 'act_step': act_step+1, 'trajectory': trajectory, 'batch_identity': identity,
                'loss': loss_value, 'preclip_norm': raw_norm.item(), 'lr': lr, 'strict_operator_correct': correct_count,
                'total': size, 'optimizer_steps': self.optimizer_steps, 'ema_steps': self.ema_steps, 'latent_displacement': trace}

    @torch.no_grad()
    def evaluate_batch(self, batch, *, act_steps=None):
        depth = self.config.act_steps if act_steps is None else act_steps
        nonrecursive = isinstance(self.model, NonrecursiveModel)
        if nonrecursive:
            require(act_steps in (None, 1), 'NR evaluation is single pass')
        else:
            require(depth in (8, 16), 'evaluation depth must be 8 or 16')
        if depth == 8 and not nonrecursive:
            return super().evaluate_batch(batch)
        original = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        was_training = self.model.training
        try:
            for name, p in self.model.named_parameters():
                p.copy_(self.ema[name])
            self.model.eval()
            if nonrecursive:
                logits = self.model(batch['token_ids'])
            else:
                carry = None
                for act_step in range(depth):
                    logits, carry = self.model.forward_act(batch['token_ids'], BackwardPolicy.for_arm('FULL'), carry=carry, act_step=act_step)
            loss, predictions, correct = supervised_loss(logits, batch)
            return {'loss': loss.item(), 'predictions': predictions.cpu(), 'strict_operator_correct': int(correct.sum()), 'total': len(correct)}
        finally:
            for name, p in self.model.named_parameters():
                p.copy_(original[name])
            self.model.train(was_training)


def tensors(rows):
    return {k: torch.tensor([row[k] for row in rows], dtype=torch.bool if k == 'target_mask' else torch.long)
            for k in ('token_ids', 'target_positions', 'target_operator_ids', 'target_mask')}


def evaluator():
    from . import evaluation
    return evaluation


def evaluate(trainer, rows, device, batch_size=512, *, act_steps=None):
    # Actual experiment final EMA evaluator: identical stratum batching, loss
    # denominators, strict canonical comparisons and secondary semantic counts.
    original = evaluator()
    depth = (1 if isinstance(trainer.model, NonrecursiveModel) else trainer.config.act_steps) if act_steps is None else act_steps
    evaluated = trainer if act_steps is None else SimpleNamespace(evaluate_batch=partial(trainer.evaluate_batch, act_steps=depth))
    result = original.evaluate_split(evaluated, rows, batch_size, device, original.histogram(rows))
    result['scope'] = f'Current EMA; full fixed ACT{depth}; experiment strict/semantic evaluator; ID only'
    return result
