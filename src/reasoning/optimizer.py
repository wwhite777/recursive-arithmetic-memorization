"""Restricted Adam-atan2; MIT source notice in third_party/ADAM_ATAN2_LICENSE.

Based on the update in lucidrains/adam-atan2-pytorch at
60ba5715f52d6b7f099bba1cba3baa1548e04410. Optional cautious/regenerative
features are excluded. a=1.27,b=1, bias correction and lr-scaled decoupled
parameter decay follow that inspected implementation; no CUDA extension.
"""
import torch


class AdamAtan2(torch.optim.Optimizer):
    def __init__(self, params, lr=5e-4, betas=(.9, .95), weight_decay=1.0, a=1.27, b=1.0):
        if lr <= 0 or weight_decay < 0 or a <= 0 or b <= 0 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError('invalid Adam-atan2 constants')
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, a=a, b=b))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None: continue
                grad = parameter.grad
                if grad.is_sparse: raise ValueError('dense optimizer only')
                state = self.state[parameter]
                if not state:
                    state.update(steps=0, exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
                state['steps'] += 1
                step = state['steps']
                state['exp_avg'].lerp_(grad, 1-beta1)
                state['exp_avg_sq'].lerp_(grad.square(), 1-beta2)
                numerator = state['exp_avg'] / (1-beta1**step)
                denominator = (state['exp_avg_sq'] * (group['b']**2 / (1-beta2**step))).sqrt()
                update = torch.atan2(numerator, denominator)
                parameter.mul_(1-group['lr']*group['weight_decay'])
                parameter.add_(update, alpha=-group['lr']*group['a'])
        return loss
