# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['DPA']

from torch import nn

from .memory import ForwardBackwardMemory
from .tau import TAU


class DPA(nn.Module):
    def __init__(self,
                 num_features: int,
                 tau: float = 0,
                 global_pool: str = 'token',
                 momentum: float = 0.9,
                 update_interval: int = 5,
                 num_memory_entries: int = 1000):
        super().__init__()

        self.tau = TAU(tau=tau)
        self.memory = ForwardBackwardMemory(num_features, momentum, update_interval, tau, num_memory_entries)

        self.tau.register_backward_hook(self._backward_hook)
        self.global_pool = global_pool

    def _backward_hook(self, module, grad_input, grad_output):
        if self.targets is not None:
            if self.global_pool == 'token':
                self.memory.update_grad(grad_output[0][:, 0], self.targets)
            elif self.global_pool == 'avg':
                self.memory.update_grad(grad_output[0].mean((-1, -2)), self.targets)
            else:
                raise ValueError

    def forward(self, x, targets=None, preact_values_list=None):
        self.targets = targets

        if targets is not None:
            if self.global_pool == 'token':
                values = x[:, 0]
                self.memory.update_value(values, targets)
            elif self.global_pool == 'avg':
                values = x.mean((-1, -2))
                self.memory.update_value(values, targets)
            else:
                raise ValueError

        x = self.tau(x)

        if targets is not None:
            neg_samples = values[self.memory.mask]
            preact_values_list.append(neg_samples)

        return x
