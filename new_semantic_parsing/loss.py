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
"""Loss functions used in the model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, preds, target, mask):
        """
        :param preds: torch.FloatTensor of shape (batch_size, seq_len, vocab_size)
        :param target: torch.LongTensor of shape (batch_size, seq_len)
        :param mask: torch.FloatTensor with 0 at masked tokens
        :return:
        """
        if mask is None:
            mask = torch.ones_like(target)

        log_probs = F.log_softmax(preds, dim=-1)

        one_hot = torch.zeros_like(preds).scatter(1, target.view(-1, 1), 1)
        one_hot = (1 - self.eps) * one_hot + self.eps / preds.size(-1) * one_hot

        loss = -(one_hot * log_probs).sum(dim=-1)
        loss *= mask
        loss = loss.sum() / torch.sum(mask)  # divide by the total number of tokens

        return loss
