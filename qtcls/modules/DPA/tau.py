# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['TAU']

import torch
from torch import nn


class CoreTAU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, tau: torch.Tensor):
        ctx.save_for_backward(input, tau)
        output = torch.where(input < tau, torch.zeros_like(input), input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, tau = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < tau] = 0
        return grad_input, None


class TAU(nn.Module):
    def __init__(self, tau: float = 0):
        super().__init__()
        if tau < 0:
            raise ValueError('tau must be non-negative')
        self.register_buffer('tau', torch.tensor(float(tau)))

    def forward(self, x):
        return CoreTAU.apply(x, self.tau)
