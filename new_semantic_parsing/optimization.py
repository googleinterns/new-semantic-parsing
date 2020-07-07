# Copyright 2020 Google LLC
# Copyright 2018 The HuggingFace Inc. team.
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
"""Utils to get optimizer and scheduler.

Optimizers have different param gropus for encoder and decoder to support
gradual unfreezing and different learning rates.
"""

from itertools import chain

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_optimizers(
    model, learning_rate, warmup_steps, num_frozen_encoder_steps, weight_decay=0, adam_eps=1e-9
):
    """Setups the optimizer and the learning rate scheduler.

    Creates optimizer which can update encoder and decoder with different learning rates
    and scheduler which increases lr for warmup_steps and does not update encoder
    for num_frozen_encoder_steps.

    Args:
        model: EncoderDecoderWPointerModel.
        learning_rate: either float or dict with keys 'encoder_lr' and 'decoder_lr'.
        warmup_steps: number of steps at the start of the training when the learning rate
            is increased from zero to learning_rate value.
        num_frozen_encoder_steps: number of steps at the start of the training when encoder weights
            are not updated.
        weight_decay: optimizer weight_decay.

    Returns:
        A tuple with two values: torch Optimizer and torch LambdaLR scheduler.
    """

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    lr = learning_rate
    if isinstance(lr, float):
        encoder_lr = decoder_lr = lr
    elif isinstance(lr, dict):
        encoder_lr = lr.get("encoder_lr", 0)
        decoder_lr = lr.get("decoder_lr", 0)
    else:
        raise ValueError("learning_rate should be either float or dict")

    # decoder parameters include prediction head and pointer network
    # optionally, they also include the module which projects encoder representations
    # into decoder-sized dimension
    to_chain = [
        model.decoder.named_parameters(),
        model.lm_head.named_parameters(),
        model.decoder_q_proj.named_parameters(),
    ]
    if model.enc_dec_proj is not None:
        to_chain.append(model.enc_dec_proj.named_parameters())

    decoder_parameters = chain(*to_chain)

    # fmt: off
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in decoder_parameters if not any(nd in n for nd in no_decay)],
            "initial_lr": decoder_lr,
            "lr": decoder_lr,
            "weight_decay": weight_decay,
            "group_type": "decoder_params",
        },
        {
            "params": [p for n, p in decoder_parameters if any(nd in n for nd in no_decay)],
            "initial_lr": decoder_lr,
            "lr": decoder_lr,
            "weight_decay": 0.0,
            "group_type": "decoder_params",
        },
    ]

    if encoder_lr > 0:
        optimizer_grouped_parameters.extend([
            {
                "params": [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "initial_lr": encoder_lr,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "group_type": "encoder_params",
            },
            {
                "params": [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "initial_lr": encoder_lr,
                "lr": encoder_lr,
                "weight_decay": 0.0,
                "group_type": "encoder_params",
            },
        ])
    # fmt: on

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, eps=adam_eps, betas=(0.9, 0.98))

    scheduler = get_noam_schedule_with_gradual_unfreezing(
        optimizer,
        num_warmup_steps=warmup_steps,
        model_size=model.decoder.config.hidden_size,
        num_frozen_encoder_steps=num_frozen_encoder_steps,
    )

    return optimizer, scheduler


def get_linear_schedule_with_warmup_and_gradual_unfreezing(
    optimizer, num_warmup_steps, num_training_steps, num_frozen_encoder_steps, last_epoch=-1,
):
    """
    :param optimizer: torch Optimizer where some param groups have 'group_type' key
        if group_type starts with 'encoder_' it will be frozen for `num_frozen_encoder_steps`
    :param num_warmup_steps: number of steps for linear warmup from 0 to optimizer.lr
    :param num_training_steps: total number of training steps, used to compute lr after the warmup
    :param num_frozen_encoder_steps: number of steps without encoder updates
    :param last_epoch: The index of last epoch. Default: -1.

    :return: LambdaLR scheduler
    """

    set_encoder_requires_grad(optimizer.param_groups, False)

    def lr_lambda(current_step):
        if current_step >= num_frozen_encoder_steps:
            set_encoder_requires_grad(optimizer.param_groups, True)

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_noam_schedule_with_gradual_unfreezing(
    optimizer, num_warmup_steps, model_size, num_frozen_encoder_steps, last_epoch=1
):
    """
    Creates a Noam (inverse square root) scheduler with linear warmup and encoder gradual unfreezing.

    :param optimizer: torch Optimizer where some param groups have 'group_type' key
        if group_type starts with 'encoder_' it will be frozen for `num_frozen_encoder_steps`
    :param num_warmup_steps: number of steps for linear warmup from 0 to optimizer.lr
    :param model_size: hidden size of the model (d_model)
    :param num_frozen_encoder_steps: number of steps without encoder updates
    :param last_epoch: The index of last epoch. Default: 1.

    :return: LambdaLR scheduler
    """
    set_encoder_requires_grad(optimizer.param_groups, False)

    def lr_lambda(current_step):
        if current_step >= num_frozen_encoder_steps:
            set_encoder_requires_grad(optimizer.param_groups, True)

        current_step = max(current_step, 1)

        scale = model_size ** -0.5 * min(
            current_step ** (-0.5), current_step * num_warmup_steps ** (-1.5)
        )
        return scale

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def set_encoder_requires_grad(param_groups, value: bool):
    for param_group in param_groups:
        group_type = param_group.get("group_type", "")
        if not group_type.startswith("encoder"):
            continue

        for param in param_group["params"]:
            if param.requires_grad is value:
                # if value has already been set
                return
            param.requires_grad = value
