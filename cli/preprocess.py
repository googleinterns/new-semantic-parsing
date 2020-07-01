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
import argparse
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
import logging
from functools import reduce
from typing import List
from os.path import join as path_join

import toml
import torch
import pandas as pd

import transformers

from tqdm import tqdm

from new_semantic_parsing import (
    utils,
    TopSchemaTokenizer,
    SAVE_FORMAT_VERSION,
)
from new_semantic_parsing.data import PointerDataset
from new_semantic_parsing.dataclasses import SchemaItem


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True,
                        help='path to TOP dataset directory')
    parser.add_argument('--text-tokenizer', required=True,
                        help='pratrained tokenizer name or path to a saved tokenizer')
    parser.add_argument('--output-dir', required=True,
                        help='directory to save preprocessed data')
    parser.add_argument('--schema-vocab',
                        help='path to schema vocab to use')
    parser.add_argument('--seed', default=34)

    args = parser.parse_args(args)
    return args


def make_dataset(filepath, text_tokenizer, schema_tokenizer):
    data = pd.read_table(filepath, names=['text', 'tokens', 'schema'])

    text_ids: List[list] = [text_tokenizer.encode(text) for text in tqdm(data.tokens)]
    # TODO: move everything below to .encode_plus (and add torchification?)
    # NOTE: slow
    text_pointer_masks: List[list] = [utils.get_src_pointer_mask(i, text_tokenizer) for i in text_ids]

    schema_ids = []
    schema_pointer_masks = []

    for i, schema in tqdm(enumerate(data.schema), total=len(data)):

        item: SchemaItem = schema_tokenizer.encode_plus(schema, text_ids[i])
        schema_ids.append(item.ids)
        schema_pointer_masks.append(item.pointer_mask)

    dataset = PointerDataset(text_ids, schema_ids, text_pointer_masks, schema_pointer_masks)
    dataset.torchify()

    return dataset


if __name__ == '__main__':
    args = parse_args()

    utils.set_seed(args.seed)

    output_dir = args.output_dir

    if output_dir.exists():
        raise ValueError(f'output_dir {output_dir.as_posix()} already exists')

    # File structure:
    # that's text\tthat 's text\t[IN:UNSUPPORTED that 's text]
    train_data = pd.read_table(path_join(args.data, 'train.tsv'), names=['text', 'tokens', 'schema'])

    logger.info('Getting schema vocabulary')

    if args.schema_vocab is None:
        schema_vocab = reduce(set.union, map(utils.get_vocab_top_schema, train_data.schema))
    else:
        with open(args.schema_vocab) as f:
            schema_vocab = f.read().split('\n')

    logger.info(f'Schema vocabulary size: {len(schema_vocab)}')

    logger.info('Building tokenizers')
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_tokenizer, use_fast=True)
    schema_tokenizer = TopSchemaTokenizer(schema_vocab, text_tokenizer)

    logger.info('Tokenizing train dataset')
    train_dataset = make_dataset(os.path.join(args.data, 'train.tsv'), text_tokenizer, schema_tokenizer)

    logger.info('Tokenizing validation and test datasets')
    valid_dataset = make_dataset(os.path.join(args.data, 'eval.tsv'), text_tokenizer, schema_tokenizer)
    test_dataset = make_dataset(os.path.join(args.data, 'test.tsv'), text_tokenizer, schema_tokenizer)

    logger.info(f'Saving everything to {output_dir.as_posix()}')
    os.makedirs(args.output_dir)

    with open(path_join(output_dir, 'args.toml'), 'w') as f:
        args_dict = {'version': SAVE_FORMAT_VERSION, **vars(args)}
        toml.dump(args_dict, f)

    # text tokenizer is saved along with schema_tokenizer
    model_type = None
    if not os.path.exists(args.text_tokenizer):
        model_type = utils.get_model_type(args.text_tokenizer)

    schema_tokenizer.save(path_join(output_dir, 'tokenizer'), encoder_model_type=model_type)

    data_state = {
        'train_dataset': train_dataset,
        'valid_dataset': valid_dataset,
        'test_dataset': test_dataset,
        'version': SAVE_FORMAT_VERSION,
    }

    torch.save(data_state, path_join(output_dir, 'data.pkl'))
