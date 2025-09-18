# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['ReLU', 'GELU']

import torch
import torch.nn.functional as F
from torch import nn


class ReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False, *args, **kwargs):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return F.gelu(input)
