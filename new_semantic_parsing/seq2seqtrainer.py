# Copyright 2020 The HuggingFace Inc. team and Google LLC
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

# This file containes a modified version of transformers.Trainer
# Trainer._prediction_loop is slightly modified to support variable-length output

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.training_args import is_tpu_available

import transformers

from new_semantic_parsing.dataclasses import Seq2SeqEvalPrediciton


if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl


logger = logging.getLogger(__name__)


class Seq2SeqTrainer(transformers.Trainer):
    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        Provides different learning rates for the encoder and decoder if args.learning_rate is a dict with
        keys 'encoder_lr' and 'decoder_lr'
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]

        lr = self.args.learning_rate
        if isinstance(lr, float):
            encoder_lr = decoder_lr = lr
        elif isinstance(lr, dict):
            encoder_lr = lr.get('encoder_lr', 0)
            decoder_lr = lr.get('decoder_lr', 0)
        else:
            raise ValueError('learning_rate should be eigher float or dict')

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.decoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "initial_lr": decoder_lr,
                "lr": decoder_lr,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.decoder.named_parameters() if any(nd in n for nd in no_decay)],
                "initial_lr": decoder_lr,
                "lr": decoder_lr,
                "weight_decay": 0.0,
            },
        ]

        if encoder_lr > 0:
            optimizer_grouped_parameters.extend([
                {
                    "params": [p for n, p in self.model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                    "initial_lr": encoder_lr,
                    "lr": encoder_lr,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                    "initial_lr": encoder_lr,
                    "lr": encoder_lr,
                    "weight_decay": 0.0,
                },
            ])

        optimizer = transformers.AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info('***** Running %s *****', description)
        logger.info('  Num examples = %d', self.num_examples(dataloader))
        logger.info('  Batch size = %d', batch_size)
        eval_losses: List[float] = []
        preds: List[torch.Tensor] = None
        label_ids: List[torch.Tensor] = None
        label_masks: List[torch.Tensor] = []
        model.eval()

        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ['labels', 'lm_labels', 'masked_lm_labels'])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if preds is None:
                    preds = list(logits.detach().unbind())
                else:
                    preds += list(logits.detach().unbind())
                if inputs.get('labels') is not None:
                    if label_ids is None:
                        label_ids = list(inputs['labels'].detach().unbind())
                    else:
                        label_ids += list(inputs['labels'].detach().unbind())

                    label_mask = inputs.get('decoder_attention_mask', None)
                    if label_mask is not None:
                        label_masks += list(label_mask.detach().unbind())

        if self.args.local_rank != -1:
            raise NotImplementedError('Variable-length output is not supported in distributed mode')
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_tpu_available():
            raise NotImplementedError('Variable-length output is not supported with TPU')
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce('eval_preds', preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce('eval_label_ids', label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = [p.cpu().numpy() for p in preds]
        if label_ids is not None:
            label_ids = [l.cpu().numpy() for l in label_ids]
        if label_masks:
            label_masks = [l.cpu().numpy() for l in label_masks]

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(
                Seq2SeqEvalPrediciton(predictions=preds, label_ids=label_ids, label_masks=label_masks)
            )
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics['eval_loss'] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith('eval_'):
                metrics[f'eval_{key}'] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
