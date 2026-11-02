"""Adapted fixed8 ACT clock, one physical batch and one update per ACT step.

No learned halting/Q loss, puzzle table, accumulation, selector or OOD choice.
The caller owns data order and frozen protocol identities.
"""
from dataclasses import dataclass, asdict
import copy
import hashlib
import math

import torch

from .model import BackwardPolicy, stochastic_plan
from .optimizer import AdamAtan2


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

    def __post_init__(self):
        if self.total_updates < 1 or self.physical_batch < 1 or self.act_steps != 8 or self.warmup_updates < 0:
            raise ValueError('positive updates/batch, exactly8 ACT, nonnegative warmup required')
        if not self.engineering_fixture and self.physical_batch != 1024:
            raise ValueError('pilot uses single physical batch1024')
        if not self.engineering_fixture and self.total_updates % 8:
            raise ValueError('final update count must complete fixed8 trajectories')
        if self.peak_lr <= 0 or self.clip_norm <= 0 or not 0 <= self.min_lr_ratio <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError('invalid optimization constants')


def supervised_loss(logits, batch):
    """Stablemax CE over all valid operator tokens in one physical batch.

    This root-adopted denominator differs from HRM's per-example normalization.
    """
    positions, targets, mask = (batch[k] for k in ('target_positions', 'target_operator_ids', 'target_mask'))
    if positions.shape != targets.shape or mask.shape != targets.shape or mask.dtype != torch.bool or positions.dtype != torch.long or targets.dtype != torch.long:
        raise ValueError('invalid operator label schema')
    if positions.shape[0] != logits.shape[0] or not mask.any(-1).all() or (positions[mask] < 0).any() or (positions[mask] >= logits.shape[1]).any() or (targets[mask] < 0).any() or (targets[mask] > 3).any():
        raise ValueError('invalid supervised operator positions')
    safe_positions = torch.where(mask, positions, 0)
    chosen = logits.gather(1, safe_positions.unsqueeze(-1).expand(-1, -1, 4)).to(torch.float64)
    # The unused positive-branch reciprocal is kept finite for safe autograd.
    positive = torch.where(chosen < 0, (1-chosen).clamp_min(1).reciprocal(), chosen+1)
    logprob = positive.log() - positive.sum(-1, keepdim=True).log()
    safe_targets = torch.where(mask, targets, 0)
    losses = -logprob.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    loss = torch.where(mask, losses, 0).sum() / mask.sum()
    predictions = chosen.argmax(-1)
    correct = ((predictions == targets) | ~mask).all(-1)
    return loss, predictions, correct


def batch_identity(batch):
    digest = hashlib.sha256()
    for name in ('token_ids', 'target_positions', 'target_operator_ids', 'target_mask'):
        tensor = batch[name].detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(bytes(tensor.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


class PilotTrainer:
    def __init__(self, model, config, arm):
        if model.config.engineering_fixture != config.engineering_fixture:
            raise ValueError('model/trainer fixture scope mismatch')
        self.model, self.config, self.arm = model, config, arm
        if arm not in ('S22', 'S13', 'Oearlylate'): raise ValueError('not an adopted pilot arm')
        self.policy = BackwardPolicy.for_arm(arm)
        self.optimizer = AdamAtan2(model.parameters(), lr=config.peak_lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        self.ema = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        self.updates = self.optimizer_steps = self.ema_steps = 0
        self.carry = self.masks = self.current_batch_identity = None

    def learning_rate(self, step):
        if step <= self.config.warmup_updates:
            return self.config.peak_lr * step / max(1, self.config.warmup_updates)
        progress = (step-self.config.warmup_updates) / max(1, self.config.total_updates-self.config.warmup_updates)
        return self.config.peak_lr * (self.config.min_lr_ratio + (1-self.config.min_lr_ratio)*.5*(1+math.cos(math.pi*progress)))

    def step(self, batch, *, collect_displacement=False):
        if self.updates >= self.config.total_updates: raise ValueError('frozen update cap reached')
        if batch['token_ids'].shape[0] != self.config.physical_batch: raise ValueError('physical batch mismatch; no accumulation')
        identity = batch_identity(batch)
        act_step = self.updates % 8
        trajectory = self.updates // 8
        if act_step == 0:
            self.carry = None
            self.current_batch_identity = identity
            self.masks = stochastic_plan(self.model.config, batch['token_ids'].shape, self.config.seed, trajectory, device=batch['token_ids'].device, dtype=getattr(torch, self.model.config.forward_dtype), training=True)
        elif identity != self.current_batch_identity:
            raise ValueError('batch must remain identical for all8 ACT updates')
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        trace = [] if collect_displacement else None
        logits, carry = self.model.forward_act(batch['token_ids'], self.policy, carry=self.carry, masks=self.masks, seed=self.config.seed, trajectory=trajectory, act_step=act_step, trace=trace, trace_values=False)
        loss, _, correct = supervised_loss(logits, batch)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite supervised loss')
        loss.backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm, error_if_nonfinite=True)
        lr = self.learning_rate(self.updates+1)
        for group in self.optimizer.param_groups: group['lr'] = lr
        self.optimizer.step(); self.optimizer_steps += 1
        with torch.no_grad():
            for name, parameter in self.model.named_parameters(): self.ema[name].lerp_(parameter, 1-self.config.ema_decay)
        self.ema_steps += 1
        self.carry = carry
        self.updates += 1
        return {'update': self.updates, 'act_step': act_step+1, 'trajectory': trajectory, 'batch_identity': identity, 'loss': loss.item(), 'preclip_norm': raw_norm.item(), 'lr': lr, 'strict_operator_correct': int(correct.sum()), 'total': len(correct), 'optimizer_steps': self.optimizer_steps, 'ema_steps': self.ema_steps, 'latent_displacement': trace}

    def state_dict(self):
        return copy.deepcopy({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'ema': self.ema, 'updates': self.updates, 'optimizer_steps': self.optimizer_steps, 'ema_steps': self.ema_steps, 'carry': self.carry, 'masks': self.masks, 'current_batch_identity': self.current_batch_identity, 'arm': self.arm, 'training_config': asdict(self.config), 'model_config': asdict(self.model.config)})

    def load_state_dict(self, state):
        if state['arm'] != self.arm or state['training_config'] != asdict(self.config) or state['model_config'] != asdict(self.model.config): raise ValueError('checkpoint configuration mismatch')
        state = copy.deepcopy(state)
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        for name in ('ema', 'updates', 'optimizer_steps', 'ema_steps', 'carry', 'masks', 'current_batch_identity'): setattr(self, name, state[name])

    @torch.no_grad()
    def evaluate_batch(self, batch):
        """Full8 fixed ACT at final EMA; training carry/RNG/clocks unchanged."""
        original = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        was_training = self.model.training
        try:
            for name, p in self.model.named_parameters(): p.copy_(self.ema[name])
            self.model.eval()
            carry = None
            for act_step in range(8): logits, carry = self.model.forward_act(batch['token_ids'], BackwardPolicy.for_arm('FULL'), carry=carry, act_step=act_step)
            loss, predictions, correct = supervised_loss(logits, batch)
            return {'loss': loss.item(), 'predictions': predictions.cpu(), 'strict_operator_correct': int(correct.sum()), 'total': len(correct)}
        finally:
            for name, p in self.model.named_parameters(): p.copy_(original[name])
            self.model.train(was_training)
