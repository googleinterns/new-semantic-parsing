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
import os
import sys
import tempfile
import argparse
import logging
from pathlib import Path

import toml
import torch
import wandb
import transformers

from new_semantic_parsing import (
    EncoderDecoderWPointerModel,
    TopSchemaTokenizer,
    Seq2SeqTrainer,
)
from new_semantic_parsing.data import Seq2SeqDataCollator, PointerDataset
from new_semantic_parsing import utils, SAVE_FORMAT_VERSION


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('train')


def parse_args(args=None):
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--warmup-steps', default=0, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--max-grad-norm', default=1.0, type=float)

    # misc
    parser.add_argument('--wandb-project', default=None)
    parser.add_argument('--no-evaluation', default=False, action='store_true')
    parser.add_argument('--log-every', default=100, type=int)

    args = parser.parse_args(args)

    # set defaults for None fields
    if (args.encoder_lr is not None) ^ (args.decoder_lr is not None):
        raise ValueError('--encoder-lr and --decoder-lr should be both specified')

    args.decoder_layers = args.decoder_layers or args.layers
    args.decoder_hidden = args.decoder_hidden or args.hidden
    args.decoder_heads = args.decoder_heads or args.heads

    if args.output_dir is None:
        args.output_dir = os.path.join('output_dir', next(tempfile._get_candidate_names()))

    return args


if __name__ == '__main__':
    args = parse_args()

    data_dir = Path(args.data_dir)

    logger.info('Loading tokenizers')
    # NOTE: change as_posix to as_windows for Windows
    schema_tokenizer = TopSchemaTokenizer.load((data_dir/'tokenizer').as_posix())
    text_tokenizer: transformers.PreTrainedTokenizer = schema_tokenizer.src_tokenizer

    logger.info('Loading data')
    datasets = torch.load(data_dir/'data.pkl')
    train_dataset: PointerDataset = datasets['train_dataset']
    eval_dataset: PointerDataset = datasets['valid_dataset']

    maximal_pointer, _ = train_dataset.get_max_len()

    try:
        with open(data_dir/'args.toml') as f:
            preprocess_args = toml.load(f)
            if preprocess_args['version'] != SAVE_FORMAT_VERSION:
                logger.warning('Binary data version differs from the current version. '
                               'May cause failing and unexpected behavior')
    except FileNotFoundError:
        preprocess_args = None

    logger.info('Creating a model')
    if args.encoder_model:
        if preprocess_args is not None and preprocess_args['text_tokenizer'] != args.encoder_model:
            logger.warning('Data may have been preprocessed with a different tokenizer')
            logger.warning(f'Preprocessing tokenizer     : {preprocess_args["text_tokenizer"]}')
            logger.warning(f'Pretrained encoder tokenizer: {args.encoder_model}')

        encoder_config = transformers.AutoConfig.from_pretrained(args.encoder_model)
        encoder_config.hidden_dropout_prob = args.dropout
        encoder_config.attention_probs_dropout_prob = args.dropout

        encoder = transformers.AutoModel.from_config(encoder_config)

        if encoder.config.vocab_size != text_tokenizer.vocab_size:
            raise ValueError('Preprocessing tokenizer and model tokenizer are not compatible')

        ffn_hidden = 4 * args.decoder_hidden if args.decoder_hidden is not None else None

        decoder_config = transformers.BertConfig(
            is_decoder=True,
            vocab_size=schema_tokenizer.vocab_size + maximal_pointer,
            hidden_size=args.decoder_hidden or encoder.config.hidden_size,
            intermediate_size=ffn_hidden or encoder.config.intermediate_size,
            num_hidden_layers=args.decoder_layers or encoder.config.num_hidden_layers,
            num_attention_heads=args.decoder_heads or encoder.config.num_attention_heads,
            pad_token_id=schema_tokenizer.pad_token_id,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder, maximal_pointer, model_args=args)

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
            maximal_pointer=maximal_pointer,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
            model_args=args,
        )

    logger.info('Starting training')
    lr = args.lr or utils.get_lr(model)

    if args.encoder_lr is not None and args.decoder_lr is not None:
        lr = {'encoder_lr': args.encoder_lr, 'decoder_lr': args.decoder_lr}

    if args.no_evaluation:
        # to get the metrics
        eval_dataset = train_dataset

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
        fp16=False,
        local_rank=-1,
    )

    collator = Seq2SeqDataCollator(text_tokenizer.pad_token_id, schema_tokenizer.pad_token_id)

    os.environ["WANDB_PROJECT"] = args.wandb_project or "new_semantic_parsing"
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=utils.compute_metrics,
    )

    wandb.config.update(args)
    if preprocess_args is not None:
        wandb.config.update({'preprocess_' + k: v for k, v in preprocess_args.items()})

    train_results = trainer.train()
    logger.info(train_results)

    trainer.save_model(args.output_dir)

    if not args.no_evaluation:
        eval_results = trainer.evaluate()
        logger.info(eval_results)

    logger.info('Training finished!')
