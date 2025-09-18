from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # TODO: Clip gradients if max_grad_norm is set
            if group['max_grad_norm'] is not None:
                grads = [p.grad for p in group['params'] if p.grad is not None]
                if grads:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in group['params'] if p.grad is not None], 
                        group['max_grad_norm']
                    )
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                # State should be stored in this dictionary
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['step'] = 0

                # TODO: Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                state['step'] += 1

                # TODO: Update first and second moments of the gradients
                state['exp_avg'] = group['betas'][0] * state['exp_avg'] + (1 - group['betas'][0]) * grad
                state['exp_avg_sq'] = group['betas'][1] * state['exp_avg_sq'] + (1 - group['betas'][1]) * (grad**2)

                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in Algorithm 2 
                # https://arxiv.org/pdf/1711.05101
                new_exp_avg = state['exp_avg']
                new_exp_avg_sq = state['exp_avg_sq']
                if group['correct_bias']:
                    new_exp_avg = state['exp_avg'] / (1 - group['betas'][0]**state['step'])
                    new_exp_avg_sq = state['exp_avg_sq'] / (1 - group['betas'][1]**state['step'])

                # TODO: Update parameters
                update = -(alpha * new_exp_avg) / (new_exp_avg_sq**0.5 + group['eps'])
                update -= alpha * group['weight_decay'] * p.data
                p.data = p.data + update

                # TODO: Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss
