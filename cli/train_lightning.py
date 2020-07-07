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

Uses Lightning, largely copies train.py.
The code is not shared for easier modification while supporting backcompatibility with train.py.
"""

import os
import sys
import json
import logging
from os.path import join as path_join

import toml
import torch
import wandb
import transformers
import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger

from new_semantic_parsing import (
    EncoderDecoderWPointerModel,
    TopSchemaTokenizer,
)
from new_semantic_parsing import utils, SAVE_FORMAT_VERSION
from new_semantic_parsing.data import PointerDataset, Seq2SeqDataCollator
from new_semantic_parsing.callbacks import TransformersModelCheckpoint
from new_semantic_parsing.lightning_module import PointerModule

from cli.train import parse_args


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    args = parse_args()

    utils.set_seed(args.seed)

    wandb_logger = WandbLogger(project=args.wandb_project, tags=args.tag)
    wandb_logger.log_hyperparams(args)

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

    # /\ /\ copy of the train.py

    lightning_module = PointerModule(
        model=model,
        schema_tokenizer=schema_tokenizer,
        train_dataset=train_dataset,
        valid_dataset=eval_dataset,
        lr=lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_eps=adam_eps,
        num_frozen_encoder_steps=args.num_frozen_encoder_steps,
        log_every=args.log_every,
    )

    wandb_logger.watch(lightning_module, log="all", log_freq=args.log_every)

    # there is a werid bug that checkpoint_callback creates checkpoints
    # in the filepath subfolder, e.g. if you specify filepath=output_dir
    # the checkpoints will be created in output_dir/..
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=path_join(args.output_dir, "pl_checkpoint.ckpt"),
        save_top_k=2,
        verbose=True,
        monitor="eval_exact_match",
        mode="max",
        prefix="",
    )

    transformer_checkpoint_callback = TransformersModelCheckpoint(args.output_dir)

    early_stopping = False
    if args.early_stopping is not None:
        early_stopping = callbacks.EarlyStopping(
            monitor="eval_exact_match",
            patience=args.early_stopping,
            strict=False,
            verbose=False,
            mode="max",
        )

    lr_logger = callbacks.LearningRateLogger()

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=args.epochs,
        gpus=args.gpus,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping,
        gradient_clip_val=args.max_grad_norm,
        precision=16 if args.fp16 else 32,
        row_log_interval=args.log_every,
        callbacks=[lr_logger, transformer_checkpoint_callback],
    )

    trainer.fit(lightning_module)

    # \/ \/ copy of the train.py

    with open(path_join(args.data_dir, "tokenizer", "config.json")) as f:
        model_type = json.load(f)["model_type"]

    schema_tokenizer.save(path_join(args.output_dir, "tokenizer"), encoder_model_type=model_type)
    logger.info(f'Tokenizer saved in {path_join(args.output_dir, "tokenizer")}')

    with open(path_join(args.output_dir, "args.toml"), "w") as f:
        args_dict = {"version": SAVE_FORMAT_VERSION, **vars(args)}
        toml.dump(args_dict, f)

    logger.info("Training finished!")

    logger.info("Generating predictions")
    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=Seq2SeqDataCollator(pad_id=text_tokenizer.pad_token_id).collate_batch,
        num_workers=8,
    )

    # TODO: hardcoded devices, move evaluation logic to PointerModule
    predictions_ids, predictions_str = utils.iterative_prediction(
        model=lightning_module.model,
        dataloader=dataloader,
        schema_tokenizer=schema_tokenizer,
        max_len=63,
        num_beams=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
