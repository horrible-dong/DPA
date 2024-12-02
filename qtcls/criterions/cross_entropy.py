# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['CrossEntropy']

from typing import List, Dict

import torch.nn.functional as F

from ._base_ import BaseCriterion
from ..utils.misc import accuracy


class CrossEntropy(BaseCriterion):
    def __init__(self, losses: List[str], weight_dict: Dict[str, float]):
        super().__init__(losses, weight_dict)

    def loss_labels(self, outputs, targets, **kwargs):
        if isinstance(outputs, dict):
            assert 'logits' in outputs.keys(), \
                f"When using 'loss_labels(self, outputs, targets, **kwargs)' in '{self.__class__.__name__}', " \
                f"if 'outputs' is a dict, 'logits' MUST be the key."
            outputs = outputs["logits"]

        loss_ce = F.cross_entropy(outputs, targets, reduction='mean')
        losses = {'loss_ce': loss_ce, 'class_error': 100 - accuracy(outputs, targets)[0]}

        return losses
