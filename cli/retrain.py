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

Similar to train_lightning.py, but loads the model and trainer from checkpoint
and uses finetune_set instead of train_set from the data.pkl
"""

import os
import sys
import shutil
import pprint
import logging
import argparse
import tempfile
import contextlib
from os.path import join as path_join

import toml
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers

import new_semantic_parsing as nsp
import new_semantic_parsing.callbacks
import new_semantic_parsing.dataclasses

from new_semantic_parsing import utils, cli_utils, optimization


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args(args=None):
    """Parses cli arguments.

    This function is shared between retrain.py and retrain_simple.py
    """
    parser = argparse.ArgumentParser()

    # fmt: off

    # data
    parser.add_argument("--data-dir", required=True,
                        help="Path to preprocess.py --save-dir containing tokenizer, "
                             "data.pkl, and args.toml")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to store checkpoints and other output files")
    parser.add_argument("--eval-data-amount", default=1., type=float,
                        help="amount of validation set to use when training. "
                             "The final evaluation will use the full dataset.")
    parser.add_argument("--new-classes", default=None,
                        help="names of classes to track")

    parser.add_argument("--new-data-amount", default=1., type=float,
                        help="Amount of new data (finetune_set) to train on, 0 < amount <= 1")
    parser.add_argument("--old-data-amount", default=0., type=float,
                        help="Amount of old data (train_set) to train on, only values from {0, 1} are supported")
    parser.add_argument("--old-data-sampling-method", default="merge_subset",
                        help="how to sample from old data")
    parser.add_argument("--average-checkpoints", default=False, action="store_true")
    parser.add_argument("--new-model-weight", default=0.5, type=float)

    # model
    parser.add_argument("--model-dir", required=True,
                        help="Model directory containing 1) checkpoint loadable via "
                             "EncoderDecoderWPointerModel.from_pretrained and "
                             "2) tokenizer directory")

    # training
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--min-epochs", default=1, type=int)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--min-steps", default=None, type=int)
    parser.add_argument("--early-stopping", default=None, type=int,
                        help="Early stopping patience. No early stopping by default.")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", default=None, type=float,
                        help="By default, checkpoint lr is used.")
    parser.add_argument("--encoder-lr", default=None, type=float,
                        help="Encoder learning rate, overrides --lr")
    parser.add_argument("--decoder-lr", default=None, type=float,
                        help="Decoder learning rate, overrides --lr")

    parser.add_argument("--weight-decay", default=None, type=float)
    parser.add_argument("--dropout", default=None, type=float,
                        help="Dropout amount for the encoder and decoder, by defalut checkpoint value is used")
    parser.add_argument("--warmup-steps", default=0, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--label-smoothing", default=None, type=float)

    # --- retrain-specific
    parser.add_argument("--move-norm", default=None, type=float,
                        help="Regularization coefficient for the distance between the initial and current network")
    parser.add_argument("--move-norm-p", default=2, type=int,
                        help="Parameter p of the L-p norm used in move-norm regularization")
    parser.add_argument("--no-opt-state", default=False, action="store_true",
                        help="Initialize optimizer state randomly instead of loading it from the trainer checkpoint")
    parser.add_argument("--no-lr-scheduler", default=False, action="store_true",
                        help="Keep learning rate constant instead of scheduling it. Only works with retrain_simple.")
    parser.add_argument("--weight-consolidation", default=None, type=float,
                        help="Weight consolidation regularization strength.")

    # --- freezing
    parser.add_argument("--freeze-encoder", default=None, type=int,
                        help="Step to freeze encoder")
    parser.add_argument("--unfreeze-encoder", default=None, type=int,
                        help="Step to unfreeze encoder")
    parser.add_argument("--freeze-decoder", default=None, type=int,
                        help="Step to freeze decoder")
    parser.add_argument("--unfreeze-decoder", default=None, type=int,
                        help="Step to unfreeze decoder")
    parser.add_argument("--freeze-head", default=None, type=int,
                        help="Step to freeze head")
    parser.add_argument("--unfreeze-head", default=None, type=int,
                        help="Step to unfreeze head")

    # misc
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--log-every", default=None, type=int)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--gpus", default=None, type=int,
                        help="Number of gpus to train the model on")
    parser.add_argument("--clean-output", default=False, action="store_true")
    parser.add_argument("--split-amount-finetune", default=None, type=float,
                        help="Only used for logging, amount of data that was removed from the training set")

    # fmt: on

    args = parser.parse_args(args)

    # set defaults for None fields

    if (args.encoder_lr is not None) ^ (args.decoder_lr is not None):
        raise ValueError("--encoder-lr and --decoder-lr should be both specified")

    if args.encoder_lr is None and args.lr is not None:
        args.encoder_lr = args.lr
        args.decoder_lr = args.lr

    if args.lr is None and args.encoder_lr is not None:
        args.lr = {"encoder_lr": args.encoder_lr, "decoder_lr": args.decoder_lr}

    args.wandb_project = args.wandb_project or "new_semantic_parsing"
    args.tags = args.tags.split(",") if args.tags else []  # list is required by wandb interface
    args.new_classes = args.new_classes.split(",") if args.new_classes else []

    if args.split_amount_finetune is not None:
        args.split_amount_train = 1.0 - args.split_amount_finetune

    if args.gpus is None:
        args.gpus = 1 if torch.cuda.is_available() else 0

    if args.output_dir is None:
        args.output_dir = os.path.join("output_dir", next(tempfile._get_candidate_names()))

    if not (0 < args.new_data_amount <= 1):
        raise ValueError(f"--new-data-amount should be between 0 and 1 (exclusive)")

    if not (0 <= args.old_data_amount <= 1):
        raise ValueError(f"--old-data-amount should be between 0 and 1 (inclusive)")

    if not hasattr(nsp.dataclasses.SamplingMethods, args.old_data_sampling_method):
        raise ValueError(args.old_data_sampling_method)

    return args


def check_args(args):
    if args.lr is not None:
        raise ValueError(
            "Learning rate specification is not supported, use retrain_simple.py instead."
        )
    if args.no_lr_scheduler:
        raise ValueError("--no-lr-scheduler is not supported, use retrain_simple.py instead.")


def load_tokenizer(model_dir, data_dir):
    tokenizer_path1 = model_dir
    tokenizer_path2 = path_join(data_dir, "tokenizer")

    if os.path.exists(path_join(tokenizer_path1, "schema_vocab.txt")):
        schema_tokenizer = nsp.TopSchemaTokenizer.load(tokenizer_path1)
    elif os.path.exists(tokenizer_path2):
        schema_tokenizer = nsp.TopSchemaTokenizer.load(tokenizer_path2)
    else:
        raise ValueError("Tokenizer is not found in both --model-dir and --data-dir.")

    return schema_tokenizer


def load_data(path, new_data_amount, old_data_amount, old_data_sampling_method, wandb_logger):
    datasets = torch.load(path)
    finetune_dataset: nsp.PointerDataset = datasets["finetune_dataset"]
    if finetune_dataset is None:
        raise RuntimeError("Datafile provided does not contain finetune_dataset.")

    eval_dataset: nsp.PointerDataset = datasets["valid_dataset"]

    train_subset = finetune_dataset
    if new_data_amount is not None and new_data_amount < 1.0:
        train_subset = utils.make_subset(train_subset, new_data_amount)

    wandb_logger.log_hyperparams({"num_new_data": len(train_subset)})

    if old_data_amount > 0:
        if old_data_sampling_method == nsp.dataclasses.SamplingMethods.merge_subset:
            old_data_subset = utils.make_subset(datasets["train_dataset"], old_data_amount)
            train_subset = torch.utils.data.ConcatDataset([train_subset, old_data_subset])
        elif old_data_sampling_method == nsp.dataclasses.SamplingMethods.sample:
            train_subset = nsp.data.SampleConcatSubset(
                train_subset, datasets["train_dataset"], old_data_amount
            )
        else:
            raise ValueError(old_data_sampling_method)

    return train_subset, eval_dataset


def load_model(
    model_dir,
    dropout=None,
    move_norm=None,
    move_norm_p=None,
    label_smoothing=None,
    weight_consolidation=None,
):
    """Load a trained model and override some model properties if specified."""
    model_config = nsp.EncoderDecoderWPointerConfig.from_pretrained(model_dir)
    if dropout is not None:
        model_config.set_dropout(dropout)
    if move_norm is not None:
        model_config.move_norm = move_norm
    if move_norm_p is not None:
        model_config.move_norm_p = move_norm_p
    if label_smoothing is not None:
        model_config.label_smoothing = label_smoothing
    if weight_consolidation is not None:
        model_config.weight_consolidation = weight_consolidation

    model = nsp.EncoderDecoderWPointerModel.from_pretrained(model_dir, config=model_config)
    model.reset_initial_params()
    return model


def load_trainer(checkpoint_path, args, wandb_logger):
    """Load lightning Trainer from a checkpoint.

    Restores optimizer and scheduler states, resets epoch and freezing schedule.
    Parameters such as gradient_clip_val, gpus, accumulate_grad_batches, and reload_dataloaders_every_epoch
    are overloaded if specified in args.

    Args:
        checkpoint_path: str, initialize Trainer from this checkpoint
        args: argparse object from parse_args()
        wandb_logger: lightning WandbLogger object
    """
    # there is a werid bug that checkpoint_callback creates checkpoints
    # in the filepath subfolder, e.g. if you specify filepath=output_dir
    # the checkpoints will be created in output_dir/..
    # NOTE: we need save_top_k=1 fot checkpoint_callback.last_checkpoint_path
    # to point to the best model
    checkpoint_callback = nsp.callbacks.TransformersModelCheckpoint(
        filepath=path_join(args.output_dir, "pl_checkpoint.ckpt"),
        save_top_k=1,
        verbose=False,
        monitor="eval_exact_match",
        mode="max",
        prefix="",
    )

    early_stopping = False
    if args.early_stopping is not None:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="eval_exact_match",
            patience=args.early_stopping,
            strict=False,
            verbose=False,
            mode="max",
        )

    lr_logger = pl.callbacks.LearningRateLogger()

    # overload some of the Trainer arguments if they are provided to the script
    reload = args.old_data_sampling_method == nsp.dataclasses.SamplingMethods.sample
    trainer_kwargs = {
        "gradient_clip_val": args.max_grad_norm,
        "gpus": args.gpus,
        "accumulate_grad_batches": args.gradient_accumulation_steps,
        "reload_dataloaders_every_epoch": reload,
    }
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

    trainer = pl.Trainer(
        resume_from_checkpoint=checkpoint_path,
        logger=wandb_logger,
        max_epochs=args.epochs,
        min_epochs=args.min_epochs,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping,
        callbacks=[lr_logger],
        row_log_interval=1,
        limit_val_batches=args.eval_data_amount,
        **trainer_kwargs,
    )

    return trainer


def modify_checkpoint_for_retraining(
    checkpoint_path,
    output_dir,
    lr=None,
    weight_decay=None,
    no_opt_state=False,
    lightning_module=None,
):
    """Loads checkpoint file, modify it and save a modified checkpoint to output_dir

    Args:
        checkpoint_path: str, path to lightning checkpoint
        output_dir: str, directory to save new checkpoint with name initial_checkpoint.pl
        lr: dict, {'encoder_lr': float, 'decoder_lr': float}
        weight_decay: float
        no_opt_state: bool, whether to restore optimizer state or not
        lightning_module: PointerModule

    Returns:
        A new checkpoint path
    """
    # A trick to start training from the global_step=0
    # when still getting optimizer state and scheduler state restored
    checkpoint = torch.load(checkpoint_path)

    if no_opt_state:
        optimizer = optimization.get_optimizers(
            model=lightning_module.model,
            learning_rate=lr or lightning_module.lr,
            weight_decay=lightning_module.weight_decay,
        )
        checkpoint["optimizer_states"] = [optimizer.state_dict()]

    # global_step will be incremented in .test call
    # -1 is used to get metrics before the training
    checkpoint["global_step"] = -1
    checkpoint["epoch"] = -1

    if weight_decay is not None:
        # PointerModule has a single optimizer
        optimization.set_weight_decay(
            checkpoint["optimizer_states"][0]["param_groups"], weight_decay
        )

    os.makedirs(output_dir)
    initial_checkpoint_path = path_join(output_dir, "initial_checkpoint.pl")
    torch.save(checkpoint, initial_checkpoint_path)
    return initial_checkpoint_path


def load_lightning_module(
    checkpoint_path: str,
    model: nsp.EncoderDecoderWPointerModel,
    train_dataset: nsp.PointerDataset,
    eval_dataset: nsp.PointerDataset,
    schema_tokenizer: nsp.TopSchemaTokenizer,
    args: argparse.Namespace,
    wandb_logger: pl.loggers.WandbLogger,
):
    """Loads lightning module with some of the parameters overwritten."""

    wandb_logger.log_hyperparams({"new_classes": " ".join(args.new_classes)})

    # Lightning loads all params which are not specified in .load_from_checkpoint
    # thus, some arguments are only provided if we want to override the loaded values
    module_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "log_every": args.log_every,
        "batch_size": args.batch_size,
    }
    module_kwargs = {k: v for k, v in module_kwargs.items() if v is not None}

    # always overwrite freezing schedule because global_step starts from zero
    module_kwargs["freezing_schedule"] = nsp.dataclasses.EncDecFreezingSchedule.from_args(args)

    lightning_module = nsp.PointerModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        schema_tokenizer=schema_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        monitor_classes=args.new_classes,
        **module_kwargs,
    )

    return lightning_module


def average_checkpoints(new_model, args, save_to):
    old_model = load_model(
        args.model_dir, args.dropout, args.move_norm, args.move_norm_p, args.label_smoothing
    )
    new_model = cli_utils.average_models(
        old_model=old_model,
        new_model=new_model,
        new_model_weight=args.new_model_weight,
    )
    new_model.save_pretrained(save_to)


def main(args):
    utils.set_seed(args.seed)

    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    wandb_logger = pl.loggers.WandbLogger(project=args.wandb_project, tags=args.tags)
    wandb_logger.log_hyperparams(args)

    logger.info(f"Starting finetuning with args: \n{pprint.pformat(vars(args))}")

    logger.info("Loading tokenizers")
    schema_tokenizer = load_tokenizer(args.model_dir, args.data_dir)

    logger.info("Loading data")
    train_dataset, eval_dataset = load_data(
        path=path_join(args.data_dir, "data.pkl"),
        new_data_amount=args.new_data_amount,
        old_data_amount=args.old_data_amount,
        old_data_sampling_method=args.old_data_sampling_method,
        wandb_logger=wandb_logger,
    )
    train_args = cli_utils.load_saved_args(path_join(args.model_dir, "args.toml"))

    # NOTE: do not log metrics as hyperparameters
    wandb_logger.log_hyperparams(
        {"pretrain_" + k: v for k, v in train_args.items() if k != "metrics"}
    )

    wandb_logger.log_hyperparams({"num_total_data": len(train_dataset)})

    logger.info("Loading model")
    model = load_model(
        model_dir=args.model_dir,
        dropout=args.dropout,
        move_norm=args.move_norm,
        move_norm_p=args.move_norm_p,
        label_smoothing=args.label_smoothing,
        weight_consolidation=args.weight_consolidation,
    )

    logger.info("Preparing for training")

    # NOTE: load_lightning_module loads a different checkpoint, we probably need to deal with that
    lightning_module = load_lightning_module(
        checkpoint_path=train_args["pl_checkpoint_path"],
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        schema_tokenizer=schema_tokenizer,
        args=args,
        wandb_logger=wandb_logger,
    )

    # override some of the parameters saved in the Trainer
    checkpoint_path = modify_checkpoint_for_retraining(
        train_args["pl_checkpoint_path"],
        args.output_dir,
        args.lr,
        args.weight_decay,
        args.no_opt_state,
        lightning_module,
    )
    trainer = load_trainer(checkpoint_path, args, wandb_logger)

    # get evaluation metrics of the initial model
    pretrain_metrics = train_args["metrics"]
    _first_step_metrics = {
        "epoch": -1,
        "global_step": -1,
        **pretrain_metrics["means"],
        **pretrain_metrics["stdevs"],
    }
    wandb_logger.log_metrics(_first_step_metrics, step=-1)

    wandb_logger.watch(lightning_module, log="all", log_freq=lightning_module.log_every)

    # --- FIT

    # call .test to load optimizer state
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        trainer.test(lightning_module, lightning_module.val_dataloader(subset_size=0.01))

    cli_utils.check_config(lightning_module, trainer, args, strict=True)
    # NOTE: trainer.test loads a different checkpoint, we probably need to deal with that
    model.reset_initial_params()

    if model.config.move_norm is not None:
        assert torch.allclose(model.get_move_norm(), torch.zeros(1, device=model.device))

    trainer.fit(lightning_module)

    cli_utils.check_config(lightning_module, trainer, args, strict=True)

    with open(path_join(args.output_dir, "args.toml"), "w") as f:
        args_dict = {"version": nsp.SAVE_FORMAT_VERSION, **vars(args)}
        toml.dump(args_dict, f)

    logger.info("Training finished!")

    best_model_checkpoint = trainer.checkpoint_callback.last_checkpoint_path
    if args.average_checkpoints:
        average_checkpoints(model, args, save_to=os.path.dirname(best_model_checkpoint))

    eval_dataloader = nsp.data.make_dataloader(eval_dataset, args.batch_size, schema_tokenizer.pad_token_id)
    final_metrics, description = cli_utils.evaluate_model(
        best_model_checkpoint,
        schema_tokenizer,
        eval_dataloader,
        prefix="eval",
        max_len=train_args.get("max_tgt_len", 68),  # 68 is max_tgt_len for TOP
    )

    logger.info(description)
    wandb_logger.log_metrics({**final_metrics["means"], **final_metrics["stdevs"]})

    # Compute RI and RD
    class_weights = eval_dataset.get_class_frequencies(schema_tokenizer)
    class_weights = {f"cls/eval_{cls}_tree_path_f1": p for cls, p in class_weights.items()}

    finetuning_metrics = cli_utils.evaluate_finetuning_procedure(
        pretrain_metrics, final_metrics, class_weights
    )
    wandb_logger.log_metrics(finetuning_metrics)

    # Compute RI and RD with very small outliers stuff
    finetuning_metrics0 = cli_utils.evaluate_finetuning_procedure(
        pretrain_metrics, final_metrics, class_weights, sigma=0.1
    )
    finetuning_metrics0 = {k + "_0.1": v for k, v in finetuning_metrics0.items()}
    wandb_logger.log_metrics(finetuning_metrics0)

    wandb_logger.close()

    if args.clean_output:
        shutil.rmtree(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    check_args(args)
    main(args)
