# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

from __future__ import annotations

from collections import OrderedDict
import torch
import torch.fx as fx
from typing import NamedTuple, List, Optional


class Label(NamedTuple):
    true:      int
    predicted: int


class Evaluation(OrderedDict):

    def __setitem__(self, input_id: int, label: Label):
        if not isinstance(input_id, int):
            raise TypeError
        if not isinstance(label, Label):
            raise TypeError

        super(Evaluation, self).__setitem__(input_id, label)

    @property
    def correct(self) -> int:
        return sum((label.true == label.predicted) for label in self.values())

    @property
    def accuracy(self) -> float:
        return 100.0 * (float(self.correct) / len(self))

    def compare(self, other: Evaluation) -> float:
        """Return the percentage of matching predictions."""

        if len(set(self.keys()).symmetric_difference(set(other.keys()))) > 0:
            raise ValueError  # can only compare evaluations carried out on the same data points

        # else, I proceed with the comparison
        matched: int = 0
        for input_id, label in self.items():
            other_label = other[input_id]
            if label.predicted == other_label.predicted:
                matched += 1

        return 100.0 * (float(matched) / len(self))


def evaluate_network(network:    fx.GraphModule,
                     dataset:    torch.utils.data.Dataset,
                     integerise: bool = False,
                     indices:    Optional[List[int]] = None) -> Evaluation:

    if integerise and not hasattr(dataset, 'scale'):
        raise ValueError  # `dataset` must carry scale information to integerise inputs

    if indices is None:
        indices = list(range(0, len(dataset)))

    evaluation = Evaluation()

    for i in indices:
        x, y_true = dataset.__getitem__(i)
        x = x.unsqueeze(0)
        if integerise:
            x = (x / dataset.scale).floor()
        y_pred = int(torch.argmax(network(x), dim=1))
        evaluation[i] = Label(y_true, y_pred)

    return evaluation
