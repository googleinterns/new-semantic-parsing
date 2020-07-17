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
"""Finetune a trained model on a dataset.

Similar to train_lightning.py, but loads the model from checkpoint instead of
creating it and preprocesses the data.
"""

import os
import sys
import logging
import argparse
import tempfile
from os.path import join as path_join

import toml
import torch
import transformers
import numpy as np

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


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off

    # data
    parser.add_argument('--data-dir', required=True,
                        help='Path to preprocess.py --save-dir containing tokenizer, '
                             'data.pkl, and args.toml')
    parser.add_argument('--output-dir', default=None,
                        help='directory to store checkpoints and other output files')
    parser.add_argument('--new-classes-file', default=None,
                        help='path to a text file with names of classes to track, one class per line')
    parser.add_argument('--new-data-amount', default=1., type=float,
                        help='amount of old data (train_dataset) to train on, 0 < amount <= 1')
    parser.add_argument('--eval-data-amount', default=1., type=float,
                        help='amount of validation set to use when training. '
                             'The final evaluation will use the full dataset.')

    # model
    parser.add_argument('--model-dir', required=True,
                        help='model directory containing 1) checkpoint loadable via '
                             'EncoderDecoderWPointerModel.from_pretrained and '
                             '2) tokenizer directory')

    # training
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--early-stopping', default=None, type=int,
                        help='Early stopping patience. No early stopping by default.')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--lr', default=None, type=float,
                        help='By default, checkpoint lr is used.')
    parser.add_argument('--encoder-lr', default=None, type=float,
                        help='Encoder learning rate, overrides --lr')
    parser.add_argument('--decoder-lr', default=None, type=float,
                        help='Decoder learning rate, overrides --lr')
    parser.add_argument('--weight-decay', default=None, type=float)
    parser.add_argument('--dropout', default=None, type=float,
                        help='dropout amount for the encoder and decoder, by defalut checkpoint value is used')
    parser.add_argument('--warmup-steps', default=None, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=None, type=int)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)
    parser.add_argument('--num-frozen-encoder-steps', default=None, type=int,
                        help='number of steps with encoder weights not being updated')
    parser.add_argument('--label-smoothing', default=None, type=float)

    # misc
    parser.add_argument('--wandb-project', default=None)
    parser.add_argument('--log-every', default=None, type=int)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--gpus', default=None, type=int,
                        help='Number of gpus to train the model on')

    # fmt: on

    args = parser.parse_args(args)

    # set defaults for None fields

    if (args.encoder_lr is not None) ^ (args.decoder_lr is not None):
        raise ValueError("--encoder-lr and --decoder-lr should be both specified")

    if args.encoder_lr is not None:
        args.lr = {"encoder_lr": args.encoder_lr, "decoder_lr": args.decoder_lr}

    args.wandb_project = args.wandb_project or "new_semantic_parsing"
    args.tag = [args.tag] if args.tag else []  # list is required by wandb interface

    if args.gpus is None:
        args.gpus = 1 if torch.cuda.is_available() else 0

    if args.output_dir is None:
        args.output_dir = os.path.join("output_dir", next(tempfile._get_candidate_names()))

    if not (0 < args.new_data_amount <= 1):
        raise ValueError(f"--new-data-amount should be between 0 and 1")

    return args


def main(args):
    utils.set_seed(args.seed)

    wandb_logger = WandbLogger(project=args.wandb_project, tags=args.tag)
    wandb_logger.log_hyperparams(args)

    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    logger.info("Loading tokenizers")

    tokenizer_path1 = args.model_dir
    tokenizer_path2 = path_join(args.data_dir, "tokenizer")

    if os.path.exists(path_join(tokenizer_path1, "schema_vocab.txt")):
        schema_tokenizer = TopSchemaTokenizer.load(tokenizer_path1)
    elif os.path.exists(tokenizer_path2):
        schema_tokenizer = TopSchemaTokenizer.load(tokenizer_path2)
    else:
        raise ValueError("Tokenizer is not found in both --model-dir and --data-dir")

    text_tokenizer: transformers.PreTrainedTokenizer = schema_tokenizer.src_tokenizer

    logger.info("Loading data")

    datasets = torch.load(path_join(args.data_dir, "data.pkl"))
    train_dataset: PointerDataset = datasets["finetune_dataset"]
    if train_dataset is None:
        raise RuntimeError("Datafile provided does not contain finetune_dataset")

    eval_dataset: PointerDataset = datasets["valid_dataset"]

    if args.fp16:
        train_dataset.fp16 = True
        eval_dataset.fp16 = True

    max_src_len, _ = train_dataset.get_max_len()

    with open(path_join(args.model_dir, "args.toml")) as f:
        train_args = toml.load(f)
        if train_args["version"] != SAVE_FORMAT_VERSION:
            logger.warning(
                "Binary data version differs from the current version. "
                "May cause failing and unexpected behavior"
            )

    logger.info("Creating a model")

    model = EncoderDecoderWPointerModel.from_pretrained(args.model_dir)

    new_classes = None
    if args.new_classes_file is not None:
        with open(args.new_classes_file) as f:
            new_classes = f.read().strip().split("\n")
            wandb_logger.log_hyperparams({"new_classes": " ".join(new_classes)})

    logger.info("Starting training")

    train_subset = train_dataset
    if args.new_data_amount is not None and args.new_data_amount < 1.0:
        train_subset_size = int(args.new_data_amount * len(train_dataset))
        train_subset_ids = np.random.permutation(len(train_dataset))[:train_subset_size]
        train_subset = torch.utils.data.Subset(train_dataset, indices=train_subset_ids)

    # Lightning loads all params which are not specified in .load_from_checkpoint
    # thus, some arguments are only provided if we want to override the loaded values
    module_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "log_every": args.log_every,
        "batch_size": args.batch_size,
    }
    module_kwargs = {k: v for k, v in module_kwargs.items() if v is not None}

    lightning_module = PointerModule.load_from_checkpoint(
        train_args["pl_checkpoint_path"],
        model=model,
        schema_tokenizer=schema_tokenizer,
        train_dataset=train_subset,
        eval_dataset=eval_dataset,
        monitor_classes=new_classes,
        **module_kwargs,
    )

    # there is a werid bug (feature?) that checkpoint_callback creates checkpoints
    # in the filepath subfolder, e.g. if you specify filepath=output_dir
    # the checkpoints will be created in output_dir/..
    checkpoint_callback = TransformersModelCheckpoint(
        filepath=path_join(args.output_dir, "pl_checkpoint.ckpt"),
        save_top_k=1,
        verbose=True,
        monitor="eval_exact_match",
        mode="max",
        prefix="",
    )

    early_stopping = False
    if args.early_stopping is not None:
        early_stopping = callbacks.EarlyStopping(
            monitor="eval_exact_match",
            patience=args.early_stopping,
            strict=False,
            verbose=True,
            mode="max",
        )

    lr_logger = callbacks.LearningRateLogger()

    trainer_kwargs = {
        "gradient_clip_val": args.max_grad_norm,
        "gpus": args.gpus,
        "accumulate_grad_batches": args.gradient_accumulation_steps,
    }
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

    # A trick to start training from the global_step=0
    # when still getting optimizer state and scheduler state restored
    checkpoint = torch.load(train_args["pl_checkpoint_path"])

    # global_step will be incremented in .test call
    # -1 is used to get metrics before the training
    checkpoint["global_step"] = -1
    checkpoint["epoch"] = -1

    initial_checkpoint_path = path_join(args.output_dir, "initial_checkpoint.pl")
    torch.save(checkpoint, initial_checkpoint_path)

    trainer = Trainer(
        resume_from_checkpoint=initial_checkpoint_path,
        logger=wandb_logger,
        max_epochs=args.epochs,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping,
        precision=16 if args.fp16 else 32,
        callbacks=[lr_logger],
        row_log_interval=1,
        limit_val_batches=args.eval_data_amount,
        **trainer_kwargs,
    )

    # evaluate the model before training
    out = trainer.test(lightning_module, lightning_module.val_dataloader())
    out = {"eval" + k.lstrip("test"): v for k, v in out["test_metrics"].items()}
    out["epoch"] = -1
    out["global_step"] = -1
    wandb_logger.log_metrics(out)

    wandb_logger.watch(lightning_module, log="all", log_freq=lightning_module.log_every)
    trainer.fit(lightning_module)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EncoderDecoderWPointerModel.from_pretrained(args.output_dir)
    model = model.to(device)

    # \/ \/ copy of the train.py

    with open(path_join(args.output_dir, "args.toml"), "w") as f:
        args_dict = {"version": SAVE_FORMAT_VERSION, **vars(args)}
        toml.dump(args_dict, f)

    logger.info("Training finished!")

    logger.info("Generating predictions")
    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=lightning_module.batch_size,
        collate_fn=Seq2SeqDataCollator(pad_id=text_tokenizer.pad_token_id).collate_batch,
        num_workers=8,
    )

    predictions_ids, predictions_str = utils.iterative_prediction(
        model=model,
        dataloader=dataloader,
        schema_tokenizer=schema_tokenizer,
        max_len=63,
        num_beams=1,
        device=device,
    )

    logger.info("Computing evaluation metrics on full dataset")

    targets_ids = [list(ex.labels.numpy()[:-1]) for ex in eval_dataset]

    exact_match_ids = sum(
        int(str(p) == str(l)) for p, l in zip(predictions_ids, targets_ids)
    ) / len(eval_dataset)

    logger.info(f"Exact match (ids): {exact_match_ids}")
    wandb_logger.log_metrics({"eval_exact_match": exact_match_ids})
    wandb_logger.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
