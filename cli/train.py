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
"""Train the model using preprocessed (binary) data and save the model and tokenizer into a directory.

Uses modified Transformers.Trainer. Probably will be deprecated in favor to train_lightning.py
"""

import os
import sys
import json
import tempfile
import argparse
import logging
from os.path import join as path_join

import toml
import torch
import wandb
import transformers
import pandas as pd

from new_semantic_parsing import (
    EncoderDecoderWPointerModel,
    TopSchemaTokenizer,
    Seq2SeqTrainer,
)
from new_semantic_parsing.data import Seq2SeqDataCollator, PointerDataset
from new_semantic_parsing import utils, SAVE_FORMAT_VERSION, optimization

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off

    # files
    parser.add_argument('--data-dir', required=True,
                        help='Path to preprocess.py --save-dir containing tokenizer, '
                             'data.pkl, and args.toml')
    parser.add_argument('--output-dir', default=None,
                        help='directory to store checkpoints and other output files')
    # model
    parser.add_argument('--encoder-model', default=None,
                        help='pretrained model name, e.g. bert-base-uncased')
    parser.add_argument('--layers', default=None, type=int,
                        help='number of layers in the encoder. '
                             'Only used if --encoder-model is not provided.')
    parser.add_argument('--hidden', default=None, type=int,
                        help='hidden size of the encoder. '
                             'Only used if --encoder-model is not provided.')
    parser.add_argument('--heads', default=None, type=int,
                        help='hidden size of the encoder. '
                             'Only used if --encoder-model is not provided.')
    parser.add_argument('--decoder-layers', default=None, type=int,
                        help='number of layers in the decoder. '
                             'Equal to the number of the encoder layers by default')
    parser.add_argument('--decoder-hidden', default=None, type=int,
                        help='hidden size of the decoder. '
                             'Equal to the hidden side of the encoder by default')
    parser.add_argument('--decoder-heads', default=None, type=int,
                        help='hidden size of the decoder. '
                             'Equal to the number of the encoder heads by default')

    # model architecture changes
    parser.add_argument('--use-pointer-bias', default=False, action='store_true',
                        help='Use bias in pointer network')
    parser.add_argument('--decoder-head-type', default='ffn', choices=['ffn', 'linear'],
                        help='Type of network used to make logits from the last decoder state')

    # training
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--early-stopping', default=None, type=int,
                        help='Lightning-only. Early stopping patience. No early stopping by default.')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--lr', default=None, type=float,
                        help='By default, lr is chosen according to the Scaling Laws for Neural Language Models')
    parser.add_argument('--encoder-lr', default=None, type=float,
                        help='Encoder learning rate, overrides --lr')
    parser.add_argument('--decoder-lr', default=None, type=float,
                        help='Decoder learning rate, overrides --lr')
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout amount for the encoder and decoder, default value 0.1 is from Transformers')
    parser.add_argument('--warmup-steps', default=1, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--num-frozen-encoder-steps', default=0, type=int,
                        help='number of steps with encoder weights not being updated')
    parser.add_argument('--label-smoothing', default=0.1, type=float)

    # misc
    parser.add_argument('--wandb-project', default=None)
    parser.add_argument('--log-every', default=100, type=int)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--gpus', default=None, type=int,
                        help='Lightning-only. Number of gpus to train the model on')

    # fmt: on

    args = parser.parse_args(args)

    # set defaults for None fields
    if (args.encoder_lr is not None) ^ (args.decoder_lr is not None):
        raise ValueError("--encoder-lr and --decoder-lr should be both specified")

    args.decoder_layers = args.decoder_layers or args.layers
    args.decoder_hidden = args.decoder_hidden or args.hidden
    args.decoder_heads = args.decoder_heads or args.heads
    args.wandb_project = args.wandb_project or "new_semantic_parsing"
    args.tag = [args.tag] if args.tag else []  # list is required by wandb interface

    if args.gpus is None:
        args.gpus = 1 if torch.cuda.is_available() else 0

    if args.output_dir is None:
        args.output_dir = os.path.join("output_dir", next(tempfile._get_candidate_names()))

    return args


if __name__ == "__main__":
    args = parse_args()

    wandb.init(project=args.wandb_project, config=args, tags=args.tag)

    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    logger.info("Loading tokenizers")
    schema_tokenizer = TopSchemaTokenizer.load(path_join(args.data_dir, "tokenizer"))
    text_tokenizer: transformers.PreTrainedTokenizer = schema_tokenizer.src_tokenizer

    logger.info("Loading data")
    datasets = torch.load(path_join(args.data_dir, "data.pkl"))
    train_dataset: PointerDataset = datasets["train_dataset"]
    eval_dataset: PointerDataset = datasets["valid_dataset"]

    if args.fp16:
        train_dataset.fp16 = True
        eval_dataset.fp16 = True

    max_src_len, _ = train_dataset.get_max_len()

    try:
        with open(path_join(args.data_dir, "args.toml")) as f:
            preprocess_args = toml.load(f)
            if preprocess_args["version"] != SAVE_FORMAT_VERSION:
                logger.warning(
                    "Binary data version differs from the current version. "
                    "May cause failing and unexpected behavior"
                )
            wandb.config.update({"preprocess_" + k: v for k, v in preprocess_args.items()})

    except FileNotFoundError:
        preprocess_args = None

    logger.info("Creating a model")
    if args.encoder_model:
        if preprocess_args is not None and preprocess_args["text_tokenizer"] != args.encoder_model:
            logger.warning("Data may have been preprocessed with a different tokenizer")
            logger.warning(f'Preprocessing tokenizer     : {preprocess_args["text_tokenizer"]}')
            logger.warning(f"Pretrained encoder tokenizer: {args.encoder_model}")

        encoder_config = transformers.AutoConfig.from_pretrained(args.encoder_model)
        encoder_config.hidden_dropout_prob = args.dropout
        encoder_config.attention_probs_dropout_prob = args.dropout

        encoder = transformers.AutoModel.from_pretrained(args.encoder_model, config=encoder_config)

        if encoder.config.vocab_size != text_tokenizer.vocab_size:
            raise ValueError("Preprocessing tokenizer and model tokenizer are not compatible")

        ffn_hidden = 4 * args.decoder_hidden if args.decoder_hidden is not None else None

        decoder_config = transformers.BertConfig(
            is_decoder=True,
            vocab_size=schema_tokenizer.vocab_size + max_src_len,
            hidden_size=args.decoder_hidden or encoder.config.hidden_size,
            intermediate_size=ffn_hidden or encoder.config.intermediate_size,
            num_hidden_layers=args.decoder_layers or encoder.config.num_hidden_layers,
            num_attention_heads=args.decoder_heads or encoder.config.num_attention_heads,
            pad_token_id=schema_tokenizer.pad_token_id,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(
            encoder=encoder, decoder=decoder, max_src_len=max_src_len, model_args=args,
        )

    else:  # if args.encoder_model is not specified
        model = EncoderDecoderWPointerModel.from_parameters(
            layers=args.layers,
            hidden=args.hidden,
            heads=args.heads,
            decoder_layers=args.decoder_layers,
            decoder_hidden=args.decoder_hidden,
            decoder_heads=args.decoder_heads,
            src_vocab_size=text_tokenizer.vocab_size,
            tgt_vocab_size=schema_tokenizer.vocab_size,
            encoder_pad_token_id=text_tokenizer.pad_token_id,
            decoder_pad_token_id=schema_tokenizer.pad_token_id,
            max_src_len=max_src_len,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
            model_args=args,
        )

    logger.info("Starting training")
    lr = args.lr or utils.get_lr(model)

    if args.encoder_lr is not None and args.decoder_lr is not None:
        lr = {"encoder_lr": args.encoder_lr, "decoder_lr": args.decoder_lr}

    adam_eps = 1e-7 if args.fp16 else 1e-9

    train_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.epochs,
        seed=args.seed,
        evaluate_during_training=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        logging_steps=args.log_every,
        save_steps=1000,
        save_total_limit=1,
        fp16=args.fp16,
        adam_epsilon=adam_eps,
        local_rank=-1,
    )

    collator = Seq2SeqDataCollator(text_tokenizer.pad_token_id, schema_tokenizer.pad_token_id)

    # number of batches not considering gradient accumulation
    epoch_len = len(train_dataset) // args.batch_size + int(len(train_dataset) % args.batch_size)
    optimizer_scheduler = optimization.get_optimizers(
        model=model,
        learning_rate=lr,
        warmup_steps=args.warmup_steps,
        num_frozen_encoder_steps=args.num_frozen_encoder_steps,
        weight_decay=args.weight_decay,
        adam_eps=adam_eps,
    )

    meter = utils.MetricsMeter(
        stop_token_ids=[schema_tokenizer.eos_token_id, schema_tokenizer.pad_token_id]
    )

    # Trainer wandb interface:
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_WATCH"] = "all"

    # force tensorboard off
    transformers.trainer.is_tensorboard_available = lambda: False

    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator.collate_batch,
        compute_metrics=meter.compute_metrics,
        optimizers=optimizer_scheduler,
    )

    train_results = trainer.train()
    logger.info(train_results)

    trainer.save_model(args.output_dir)

    with open(path_join(args.data_dir, "tokenizer", "config.json")) as f:
        model_type = json.load(f)["model_type"]

    schema_tokenizer.save(path_join(args.output_dir, "tokenizer"), encoder_model_type=model_type)
    logger.info(f'Tokenizer saved in {path_join(args.output_dir, "tokenizer")}')

    with open(path_join(args.output_dir, "args.toml"), "w") as f:
        args_dict = {"version": SAVE_FORMAT_VERSION, **vars(args)}
        toml.dump(args_dict, f)

    model.eval()

    eval_results = trainer.evaluate()
    logger.info("Final eval results")
    logger.info(eval_results)

    logger.info("Training finished!")

    logger.info("Generating predictions")
    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=Seq2SeqDataCollator(pad_id=text_tokenizer.pad_token_id).collate_batch,
        num_workers=8,
    )

    predictions_ids, predictions_str = utils.iterative_prediction(
        model=model,
        dataloader=dataloader,
        schema_tokenizer=schema_tokenizer,
        max_len=63,
        num_beams=1,
        device=trainer.args.device,
    )

    # Finish the script if evaluation texts are not available
    if preprocess_args is None:
        exit()

    logger.info("Computing inference-time metrics")

    data_df = pd.read_table(
        path_join(preprocess_args["data"], "eval.tsv"), names=["text", "tokens", "schema"],
    )
    targets_str = list(data_df.schema)

    predictions_str = [schema_tokenizer.postprocess(p) for p in predictions_str]
    exact_match = sum(int(p == t) for p, t in zip(predictions_str, targets_str)) / len(targets_str)
    logger.info(f"Exact match (str): {exact_match}")

    targets_ids = [list(ex.labels.numpy()[:-1]) for ex in eval_dataset]
    exact_match_ids = sum(
        int(str(p) == str(l)) for p, l in zip(predictions_ids, targets_ids)
    ) / len(targets_str)
    logger.info(f"Exact match (ids): {exact_match_ids}")

    logger.info("Checking for mismatches between ids and str")

    n_errors = 0

    for i in range(len(targets_str)):
        if (
            str(predictions_ids[i]) == str(eval_dataset[i].labels.numpy()[:-1])
            and predictions_str[i] != targets_str[i]
        ):
            n_errors += 1
            logger.info("Mismatch ", n_errors)

            logger.info("Target str: ", targets_str[i])
            logger.info("Decoded   : ", predictions_str[i])

            logger.info("Target ids : ", eval_dataset[i].labels)
            logger.info("Predictions: ", predictions_ids[i])
            logger.info("")

    if n_errors > 0:
        logger.info(f"Mismatches       : {n_errors}")
        logger.info(f"Exact match (str): {exact_match}")
        logger.info(f"Exact match (ids): {exact_match_ids}")
