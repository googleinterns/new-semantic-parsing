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

from typing import NewType, List, Union
from dataclasses import dataclass

import numpy as np
import torch


Tensor = NewType('Tensor', Union[List, np.ndarray, torch.Tensor])
LongTensor = NewType('LongTensor', Union[List, np.ndarray, torch.LongTensor])
FloatTensor = NewType('FloatTensor', Union[List, np.ndarray, torch.FloatTensor])


@dataclass
class InputDataClass:
    input_ids: LongTensor
    decoder_input_ids: LongTensor = None
    attention_mask: FloatTensor = None
    decoder_attention_mask: FloatTensor = None
    pointer_mask: FloatTensor = None
    decoder_pointer_mask: FloatTensor = None
    labels: LongTensor = None


@dataclass
class SchemaItem:
    ids: List[int]
    pointer_mask: List[int]

    def __len__(self):
        return len(self.ids)


@dataclass
class Seq2SeqEvalPrediciton:
    predictions: List[np.ndarray]
    label_ids: List[np.ndarray]
    label_masks: List[np.ndarray] = None

