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
"""LightningModule to handle training."""

from typing import Union, Dict

import torch

from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from new_semantic_parsing import utils
from new_semantic_parsing.optimization import get_optimizers
from new_semantic_parsing.data import PointerDataset, Seq2SeqDataCollator
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer
from new_semantic_parsing.modeling_encoder_decoder_wpointer import EncoderDecoderWPointerModel


class PointerModule(LightningModule):
    """Handles training.

    Creates optimizers and implements training loop with multi-GPU and TPU support.

    Attributes:
        model: EncoderDecoderWPointerModel to train
        schema_tokenizer: TopSchemaTokenizer, mainly used for getting tokenizer options like BOS/EOS/PAD token ids
        text_tokenizer: transformers.PreTrainedTOkenizer, ^^
        train_dataset: PointerDataset
        valid_dataset: PointerDataset
        test_dataset: optional, PointerDataset which may not have labels
        lr: learning rate, either float of dictionary with keys encoder_lr and decoder_lr
        batch_size: int, batch size used for training and evaluation
        warmup_steps: int
        weight_decay: float
        num_frozen_encoder_steps: int, number of steps at the beginning of the training when encoder weights are not updated
        log_every: int, log to wandb each log_every steps
    """

    def __init__(
        self,
        model: EncoderDecoderWPointerModel,
        schema_tokenizer: TopSchemaTokenizer,
        train_dataset: PointerDataset,
        valid_dataset: PointerDataset,
        lr: Union[float, Dict],
        batch_size=32,
        warmup_steps=0,
        weight_decay=0.0,
        adam_eps=1e-9,
        num_frozen_encoder_steps=0,
        test_dataset=None,
        log_every=50,
    ):
        super().__init__()
        self.model = model

        self.schema_tokenizer = schema_tokenizer
        self.text_tokenizer = schema_tokenizer.src_tokenizer

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.lr = lr
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.num_frozen_encoder_steps = num_frozen_encoder_steps
        self.adam_eps = adam_eps
        self.log_every = log_every

        self._collator = Seq2SeqDataCollator(
            pad_id=self.text_tokenizer.pad_token_id,
            decoder_pad_id=self.schema_tokenizer.pad_token_id,
        )

    def forward(self, *args, **kwargs):
        """Coincides with EncoderDecoderWPointerModel.forward"""
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]

        if batch_idx % self.log_every != 0:
            return {"loss": loss}

        logits = outputs[1]
        preds = logits.max(-1).indices

        labels = batch["labels"]
        label_masks = batch["decoder_attention_mask"]

        stop_token_ids = [self.schema_tokenizer.eos_token_id, self.schema_tokenizer.pad_token_id]
        metrics = utils.compute_metrics_from_batch(preds, labels, label_masks, stop_token_ids)

        log_dict = {
            "loss": loss,
            "train_batch_accuracy": metrics["accuracy"],
            "train_batch_exact_match": metrics["exact_match"],
        }
        return {"loss": loss, "log": log_dict}

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizers(
            model=self.model,
            learning_rate=self.lr,
            warmup_steps=self.warmup_steps,
            num_frozen_encoder_steps=self.num_frozen_encoder_steps,
            weight_decay=self.weight_decay,
            adam_eps=self.adam_eps,
        )

        # to call scheduler every step instead of every epoch
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            collate_fn=self._collator.collate_batch,
        )
        return loader

    # --- Validation

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs[0]

        logits = outputs[1]
        preds = logits.max(-1).indices

        labels = batch["labels"]
        label_masks = batch["decoder_attention_mask"]

        stop_token_ids = [self.schema_tokenizer.eos_token_id, self.schema_tokenizer.pad_token_id]
        metrics = utils.compute_metrics_from_batch(preds, labels, label_masks, stop_token_ids)

        return {
            "eval_loss": loss,
            "eval_accuracy": metrics["accuracy"],
            "eval_exact_match": metrics["exact_match"],
        }

    def validation_epoch_end(self, outputs):
        avg_em = torch.stack([x["eval_exact_match"] for x in outputs]).mean()
        avg_acc = torch.stack([x["eval_accuracy"] for x in outputs]).mean()
        avg_loss = torch.stack([x["eval_loss"] for x in outputs]).mean()

        log_dict = {"eval_exact_match": avg_em, "eval_accuracy": avg_acc, "eval_loss": avg_loss}

        return {"eval_loss": avg_loss, "log": log_dict}

    def val_dataloader(self):
        loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
            collate_fn=self._collator.collate_batch,
        )
        return loader
