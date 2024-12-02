# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['ForwardBackwardMemory']

import numpy as np
import torch
from torch import nn


class MemoryManager:
    def __init__(self, mean_value_ptr=None, mean_grad_ptr=None, momentum=0.9):
        self._mean_value = mean_value_ptr
        self._mean_grad = mean_grad_ptr
        self.momentum = momentum

    @property
    def mean_value(self):
        return self._mean_value.copy()

    @property
    def mean_grad(self):
        return self._mean_grad.copy()

    def update_value(self, new_value):
        self._mean_value += self.momentum * (new_value - self._mean_value)

    def update_grad(self, new_grad):
        self._mean_grad += self.momentum * (new_grad - self._mean_grad)


class ForwardBackwardMemory(nn.Module):
    def __init__(self,
                 num_features: int,
                 momentum: float = 0.9,
                 update_interval: int = 5,
                 tau: float = 0,
                 num_memory_entries: int = 1000):
        super().__init__()

        self.num_features = num_features

        self.value_memory = np.zeros([num_memory_entries, num_features], dtype=np.float32)  # forward memory
        self.grad_memory = np.zeros([num_memory_entries, num_features], dtype=np.float32)  # backward memory
        self.memory_managers = [MemoryManager(mean_value_ptr=self.value_memory[i],
                                              mean_grad_ptr=self.grad_memory[i],
                                              momentum=momentum)
                                for i in range(num_memory_entries)]

        self.update_interval = update_interval
        self.count = 0

        self.tau = tau

    def update_value(self,
                     values: torch.Tensor,
                     targets: torch.LongTensor):
        """
        values: torch.Tensor([B, num_features])
        targets: torch.LongTensor([B])
        """
        self.count = (self.count + 1) % self.update_interval

        assert values.shape[1] == self.num_features

        batch_mean_value = []
        batch_mean_grad = []

        for value, target in zip(values.detach().cpu().numpy().astype(np.float32), targets.tolist()):
            memory_manager = self.memory_managers[target]
            if self.training and self.count == 0:
                memory_manager.update_value(value)
            batch_mean_value.append(memory_manager.mean_value)
            batch_mean_grad.append(memory_manager.mean_grad)

        batch_mean_value = torch.from_numpy(np.stack(batch_mean_value)).to(values.device)
        batch_mean_grad = torch.from_numpy(np.stack(batch_mean_grad)).to(values.device)

        mask_value = (batch_mean_value < self.tau) & (values > self.tau)
        mask_grad = batch_mean_grad < 0

        self.mask = mask_value & mask_grad

    def update_grad(self,
                    grads: torch.Tensor,
                    targets: torch.LongTensor):
        """
        grads: torch.Tensor([B, num_features])
        targets: torch.LongTensor([B])
        """
        assert grads.shape[1] == self.num_features

        if self.training and self.count == 0:
            for grad, target in zip(torch.where(grads > 0, 1, -1).cpu().numpy().astype(np.float32), targets.tolist()):
                self.memory_managers[target].update_grad(grad)
