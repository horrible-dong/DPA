# Copyright (c) QIU Tian. All rights reserved.

import torch

from ..cross_entropy import CrossEntropy

__all__ = ['ChannelLoss']


class ChannelLoss(CrossEntropy):
    def __init__(self, losses: list, weight_dict: dict):
        super().__init__(losses, weight_dict)

    def loss_channel(self, outputs, targets, **kwargs):
        values = outputs["preact_values"]
        loss_channel = (values ** 2).mean() if len(values) > 0 else torch.tensor(0.).to(values.device)
        losses = {'loss_channel': loss_channel}
        return losses
