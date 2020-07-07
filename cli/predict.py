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
"""Make predictions using a trained model and save them into a file.

Optionally, compute metrics if the labels are available.
Only works with transformers.PreTrainedModel saved format. Lightning is not yet supported,
but train_lightning.py also saves transformers format along with lightning checkpoints.
"""

import os
import sys
import argparse
import logging
from typing import List
from os.path import join as path_join

import torch
import transformers
import pandas as pd
from tqdm.auto import tqdm

from new_semantic_parsing import (
    EncoderDecoderWPointerModel,
    TopSchemaTokenizer,
)
from new_semantic_parsing.data import Seq2SeqDataCollator, PointerDataset
from new_semantic_parsing import utils
from cli.preprocess import make_dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    parser.add_argument('--data', required=True, help='path to data file')
    parser.add_argument('--model', required=True, help='path to a model checkpoint')
    parser.add_argument('--output-file', required=True,
                        help='file to save preprocessed data')
    parser.add_argument('--schema-tokenizer', default=None,
                        help='path to a saved tokenizer (note that schema tokenizer includes text tokenizer), '
                             'by default --data/tokenizer is used')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-beams', default=4, type=int)
    parser.add_argument('--src-max-len', default=63, type=int,
                        help='maximum length of the source sequence in tokens, '
                             '63 for TOP train set and bert-base-cased tokenizer')
    parser.add_argument('--tgt-max-len', default=98, type=int,
                        help='maximum length of the target sequence in tokens, '
                             '98 for TOP train set and bert-base-cased tokenizer')
    parser.add_argument('--device', default=None,
                        help='Use CUDA if available by default')
    parser.add_argument('--seed', default=34, type=int)
    # fmt: on

    args = parser.parse_args(args)
    args.schema_tokenizer = args.schema_tokenizer or path_join(args.model, "tokenizer")

    if os.path.exists(args.output_file):
        raise ValueError(f"output file {args.output_file} already exists")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def make_test_dataset(filepath, schema_tokenizer, max_len=None):
    data = pd.read_table(filepath, names=["text", "tokens", "schema"])

    text_ids: List[list] = [
        schema_tokenizer.encode_source(text) for text in tqdm(data.tokens, desc="tokenization")
    ]
    if max_len is not None:
        text_ids = [t[:max_len] for t in text_ids]

    text_pointer_masks: List[list] = [
        utils.get_src_pointer_mask(t, text_tokenizer) for t in text_ids
    ]

    dataset = PointerDataset(source_tensors=text_ids, source_pointer_masks=text_pointer_masks)
    dataset.torchify()

    return dataset


if __name__ == "__main__":
    args = parse_args()

    logger.info("Loading tokenizers")
    schema_tokenizer = TopSchemaTokenizer.load(args.schema_tokenizer)
    text_tokenizer: transformers.PreTrainedTokenizer = schema_tokenizer.src_tokenizer

    logger.info("Loading data")

    # NOTE: this dataset object does not have labels
    dataset: PointerDataset = make_test_dataset(
        args.data, schema_tokenizer, max_len=args.src_max_len
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=Seq2SeqDataCollator(pad_id=text_tokenizer.pad_token_id).collate_batch,
        num_workers=8,
    )

    logger.info(f"Maximum source text length {dataset.get_max_len()[0]}")

    model = EncoderDecoderWPointerModel.from_pretrained(args.model).to(args.device)
    model.eval()

    predictions_ids, predictions_str = utils.iterative_prediction(
        model=model,
        dataloader=dataloader,
        schema_tokenizer=schema_tokenizer,
        max_len=args.tgt_max_len,
        num_beams=args.num_beams,
        device=args.device,
    )

    # predictions should be postprocessed for evaluation (reproduce TOP format tokenization)
    predictions_str = [schema_tokenizer.postprocess(p) for p in predictions_str]

    with open(args.output_file, "w") as f:
        for pred in predictions_str:
            f.write(pred + "\n")

    with open(args.output_file + ".ids", "w") as f:
        for pred in predictions_ids:
            f.write(str(pred) + "\n")

    logger.info(f"Prediction finished, results saved to {args.output_file}")
    logger.info(f'Ids saved to {args.output_file + ".ids"}')

    logger.info(f"Computing some metrics...")

    try:
        data_df = pd.read_table(args.data, names=["text", "tokens", "schema"])
        dataset_with_labels = make_dataset(args.data, schema_tokenizer)
        targets_str = list(data_df.schema)

        exact_match_str = sum(int(p == t) for p, t in zip(predictions_str, targets_str)) / len(
            targets_str
        )
        logger.info(f"Exact match: {exact_match_str}")

        targets_ids = [list(ex.labels.numpy()[:-1]) for ex in dataset_with_labels]
        exact_match_ids = sum(
            int(str(p) == str(l)) for p, l in zip(predictions_ids, targets_ids)
        ) / len(targets_str)
        logger.info(f"Exact match (ids): {exact_match_ids}")

    except FileNotFoundError as e:
        logger.warning(e)
