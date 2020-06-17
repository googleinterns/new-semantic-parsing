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

import transformers

from new_semantic_parsing import (
    EncoderDecoderWPointerModel,
    TopSchemaTokenizer,
    Seq2SeqTrainer,
)
from new_semantic_parsing.data import Seq2SeqDataCollator
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
                        help='number of layers in each encoder and decoder. '
                             'Ignored if --encoder-model is provided.')
    parser.add_argument('--hidden', default=None, type=int,
                        help='hidden size of the encoder and decoder. '
                             'Ignored if --encoder-model is provided')
    parser.add_argument('--heads', default=None, type=int,
                        help='hidden size of the encoder and decoder. '
                             'Ignored if --encoder-model is provided')
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
    parser.add_argument('--warmup-steps', default=0, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--wandb-project', default=None)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    if (args.encoder_lr is not None) ^ (args.decoder_lr is not None):
        raise ValueError('--encoder-lr and --decoder-lr should be both specified')

    data_dir = Path(args.data_dir)

    logger.info('Loading tokenizers')
    # NOTE: change as_posix to as_windows for Windows
    schema_tokenizer = TopSchemaTokenizer.load((data_dir/'tokenizer').as_posix())
    text_tokenizer: transformers.PreTrainedTokenizer = schema_tokenizer.src_tokenizer

    logger.info('Loading data')
    datasets = torch.load(data_dir/'data.pkl')
    train_dataset = datasets['train_dataset']
    eval_dataset = datasets['valid_dataset']

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

        encoder = transformers.AutoModel.from_pretrained(args.encoder_model)
        if encoder.config.vocab_size != text_tokenizer.vocab_size:
            raise ValueError('Preprocessing tokenizer and model tokenizer are not compatible')

        decoder_config = transformers.BertConfig(
            is_decoder=True,
            vocab_size=schema_tokenizer.vocab_size + encoder.config.vocab_size,
            hidden_size=encoder.config.hidden_size,
            intermediate_size=encoder.config.intermediate_size,
            num_hidden_layers=encoder.config.num_hidden_layers,
            num_attention_heads=encoder.config.num_attention_heads,
            pad_token_id=schema_tokenizer.pad_token_id,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder)
    else:
        model = EncoderDecoderWPointerModel.from_parameters(
            layers=args.layers,
            hidden=args.hidden,
            heads=args.heads,
            src_vocab_size=text_tokenizer.vocab_size,
            tgt_vocab_size=schema_tokenizer.vocab_size,
            encoder_pad_token_id=text_tokenizer.pad_token_id,
            decoder_pad_token_id=schema_tokenizer.pad_token_id,
        )

    logger.info('Starting training')
    lr = args.lr or utils.get_lr(model)

    if args.encoder_lr is not None and args.decoder_lr is not None:
        lr = {'encoder_lr': args.encoder_lr, 'decoder_lr': args.decoder_lr}

    if args.output_dir is None:
        args.output_dir = os.path.join('output_dir', next(tempfile._get_candidate_names()))

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
        warmup_steps=args.warmup_steps,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
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

    train_results = trainer.train()
    logger.info(train_results)

    trainer.save_model(args.output_dir)

    eval_results = trainer.evaluate()
    logger.info(eval_results)

    logger.info('Training finished!')
