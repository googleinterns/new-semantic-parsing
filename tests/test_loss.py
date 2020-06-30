# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import unittest

import torch
import torch.nn.functional as F

from new_semantic_parsing.utils import set_seed
from new_semantic_parsing.loss import LabelSmoothedCrossEntropy


class LabelSmoothedCrossEntropyTest(unittest.TestCase):
    def test_no_smoothing(self):
        set_seed(98)

        v_size = 43

        preds = torch.randn(size=(7, 19, v_size)).view(-1, v_size)
        labels = torch.randint(43, size=(7, 19)).view(-1)
        mask = torch.ones_like(labels)

        ce1 = F.cross_entropy(preds, labels)
        ce2 = LabelSmoothedCrossEntropy(eps=0)(preds, labels, mask)

        self.assertTrue(torch.allclose(ce1, ce2))

    def test_smoothing(self):
        # only checks that it does not fail
        set_seed(98)

        v_size = 43
        eps = 0.1

        preds = torch.randn(size=(7, 19, v_size)).view(-1, v_size)
        labels = torch.randint(43, size=(7, 19)).view(-1)
        mask = torch.ones_like(labels)

        ce2 = LabelSmoothedCrossEntropy(eps=eps)(preds, labels, mask)
