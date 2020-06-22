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
import shutil
import tempfile
import random
import unittest
from pprint import pprint

import torch
import transformers
import numpy as np

from new_semantic_parsing import EncoderDecoderWPointerModel, Seq2SeqTrainer
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer
from new_semantic_parsing.utils import compute_metrics, get_src_pointer_mask
from new_semantic_parsing.data import PointerDataset, Seq2SeqDataCollator


class EncoderDecoderWPointerTest(unittest.TestCase):
    def test_shape_on_random_data(self):
        torch.manual_seed(42)

        bs = 3
        src_len = 5
        tgt_len = 7

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=17,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        # decoder accepts vocabulary of schema vocab + pointer embeddings
        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=23,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        # logits are projected into schema vocab and combined with pointer scores
        max_pointer = src_len + 3
        model = EncoderDecoderWPointerModel(encoder, decoder, maximal_pointer=max_pointer)

        x_enc = torch.randint(0, encoder_config.vocab_size, size=(bs, src_len))
        x_dec = torch.randint(0, decoder_config.vocab_size, size=(bs, tgt_len))

        out = model(input_ids=x_enc, decoder_input_ids=x_dec)

        # different encoders return different number of outputs
        # e.g. BERT returns two, but DistillBERT only one
        self.assertGreaterEqual(len(out), 4)

        schema_vocab = decoder_config.vocab_size - max_pointer

        combined_logits = out[0]
        expected_shape = (bs, tgt_len, schema_vocab + src_len)
        self.assertEqual(combined_logits.shape, expected_shape)

        decoder_hidden = out[1]
        expected_shape = (bs, tgt_len, decoder_config.hidden_size)
        self.assertEqual(decoder_hidden.shape, expected_shape)

        combined_logits = out[2]
        expected_shape = (bs, decoder_config.hidden_size)
        self.assertEqual(combined_logits.shape, expected_shape)

        encoder_hidden = out[3]
        expected_shape = (bs, src_len, encoder_config.hidden_size)
        self.assertEqual(encoder_hidden.shape, expected_shape)

    def test_shape_on_real_data(self):
        torch.manual_seed(42)
        src_vocab_size = 17
        tgt_vocab_size = 23
        max_position = 5

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=src_vocab_size,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + max_position,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder, max_position)

        # similar to real data
        # e.g. '[CLS] Directions to Lowell [SEP]'
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2]])
        # e.g. '[IN:GET_DIRECTIONS Directions to [SL:DESTINATION Lowell]]'
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq,
                                decoder_input_ids=tgt_seq,
                                pointer_mask=mask)[0]

        expected_shape = (1, tgt_seq.shape[1], tgt_vocab_size + src_seq.shape[1])
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_shape_on_real_data_batched(self):
        torch.manual_seed(42)
        src_vocab_size = 17
        tgt_vocab_size = 23
        max_position = 7

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=src_vocab_size,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + max_position,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder, max_position)

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0],
                                    [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12,  7, 7, 0, 0],
                                    [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0],
                                  [0, 1, 1, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq,
                                decoder_input_ids=tgt_seq,
                                pointer_mask=mask)[0]

        expected_shape = (2, tgt_seq.shape[1], tgt_vocab_size + src_seq.shape[1])
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_loss_computation(self):
        torch.manual_seed(42)
        src_vocab_size = 17
        tgt_vocab_size = 23

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=src_vocab_size,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        max_position = 7
        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + max_position,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder, maximal_pointer=7)

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0],
                                    [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12,  7, 7, 0, 0],
                                    [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0],
                                  [0, 1, 1, 1, 1, 1, 0]])

        loss = model(input_ids=src_seq,
                     decoder_input_ids=tgt_seq,
                     pointer_mask=mask,
                     labels=tgt_seq)[0]

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertGreater(loss, 0)


class ModelOverfitTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = next(tempfile._get_candidate_names())

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        if os.path.exists('runs'):
            shutil.rmtree('runs')

    def _prepare_data(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')

        vocab = {'[', ']', 'IN:', 'SL:', 'GET_DIRECTIONS', 'DESTINATION',
                 'DATE_TIME_DEPARTURE', 'GET_ESTIMATED_ARRIVAL'}
        schema_tokenizer = TopSchemaTokenizer(vocab, tokenizer)

        source_texts = [
            'Directions to Lowell',
            'Get directions to Mountain View',
        ]
        schema_texts = [
            '[IN:GET_DIRECTIONS Directions to [SL:DESTINATION Lowell]]',
            '[IN:GET_DIRECTIONS Get directions to [SL:DESTINATION Mountain View]]'
        ]

        source_ids = [tokenizer.encode(t) for t in source_texts]
        source_pointer_masks = [get_src_pointer_mask(i, tokenizer) for i in source_ids]

        schema_ids = []
        schema_pointer_masks = []

        for src_id, schema in zip(source_ids, schema_texts):
            item = schema_tokenizer.encode_plus(schema, src_id)
            schema_ids.append(item.ids)
            schema_pointer_masks.append(item.pointer_mask)

        dataset = PointerDataset(source_ids, schema_ids, source_pointer_masks, schema_pointer_masks)
        dataset.torchify()

        return dataset, tokenizer, schema_tokenizer

    def test_overfit(self):
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        # NOTE: slow test

        dataset, tokenizer, schema_tokenizer = self._prepare_data()

        src_maxlen, _ = dataset.get_max_len()

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=3, hidden=128, heads=2, maximal_pointer=src_maxlen,
            src_vocab_size=tokenizer.vocab_size, tgt_vocab_size=schema_tokenizer.vocab_size
        )

        train_args = transformers.TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            num_train_epochs=30,
            seed=42,
            learning_rate=1e-3,
        )

        # doesn't work, need to patch transformers
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_WATCH"] = "false"
        transformers.trainer.is_wandb_available = lambda: False  # workaround

        trainer = Seq2SeqTrainer(
            model,
            train_args,
            train_dataset=dataset,
            data_collator=Seq2SeqDataCollator(model.encoder.embeddings.word_embeddings.padding_idx),
            eval_dataset=dataset,
            compute_metrics=compute_metrics,
        )
        # a trick to reduce the amount of logging
        trainer.is_local_master = lambda: False

        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        train_out = trainer.train()
        eval_out = trainer.evaluate()

        pprint('Training output')
        pprint(train_out)
        pprint('Evaluation output')
        pprint(eval_out)

        # accuracy should be 1.0 and eval loss should be around 0.9
        self.assertGreater(eval_out['eval_accuracy'], 0.99)

    def test_overfit_bert(self):
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        # NOTE: very slow test

        dataset, tokenizer, schema_tokenizer = self._prepare_data()

        src_maxlen, _ = dataset.get_max_len()

        encoder = transformers.AutoModel.from_pretrained('bert-base-cased')

        decoder = transformers.BertModel(transformers.BertConfig(
            is_decoder=True,
            vocab_size=schema_tokenizer.vocab_size + src_maxlen,
            hidden_size=encoder.config.hidden_size,
            intermediate_size=encoder.config.intermediate_size,
            num_hidden_layers=encoder.config.num_hidden_layers,
            num_attention_heads=encoder.config.num_attention_heads,
            pad_token_id=schema_tokenizer.pad_token_id,
        ))

        model = EncoderDecoderWPointerModel(encoder, decoder, src_maxlen)

        train_args = transformers.TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            num_train_epochs=50,
            seed=42,
            learning_rate=1e-4,
        )

        # doesn't work, need to patch transformers
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_WATCH"] = "false"
        transformers.trainer.is_wandb_available = lambda: False  # workaround

        trainer = Seq2SeqTrainer(
            model,
            train_args,
            train_dataset=dataset,
            data_collator=Seq2SeqDataCollator(model.encoder.embeddings.word_embeddings.padding_idx),
            eval_dataset=dataset,
            compute_metrics=compute_metrics,
        )
        # a trick to reduce the amount of logging
        trainer.is_local_master = lambda: False

        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        train_out = trainer.train()
        eval_out = trainer.evaluate()

        pprint('Training output')
        pprint(train_out)
        pprint('Evaluation output')
        pprint(eval_out)

        # accuracy should be 1.0 and eval loss should be around 0.9
        self.assertGreater(eval_out['eval_accuracy'], 0.99)
