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
"""Lightning Callbacks used when training."""

import os
from pytorch_lightning import callbacks

from new_semantic_parsing.lightning_module import PointerModule


class TransformersModelCheckpoint(callbacks.Callback):
    """Saves model in Transformers format.

    This way it is possible to simply load the model (without training hparameters)
    using transformers.from_pretrained.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_epoch_end(self, trainer, pl_module: PointerModule):
        """Saves model on epoch eng. Overrrides the previous checkpoint."""
        pl_module.model.save_pretrained(self.output_dir)
        # TODO: hardcode, encoder_model_type
        pl_module.schema_tokenizer.save(
            os.path.join(self.output_dir, "tokenizer"), encoder_model_type="bert"
        )
