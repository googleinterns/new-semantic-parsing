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
from new_semantic_parsing.utils import MetricsMeter, get_src_pointer_mask, set_seed
from new_semantic_parsing.data import PointerDataset, Seq2SeqDataCollator


MODELS = ["bert-base-cased"]


class EncoderDecoderWPointerTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = next(tempfile._get_candidate_names())

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        if os.path.exists("runs"):
            shutil.rmtree("runs")

    def test_shape_on_random_data(self):
        set_seed(42)

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
        model = EncoderDecoderWPointerModel(
            encoder=encoder, decoder=decoder, max_src_len=max_pointer
        )

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
        set_seed(42)
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

        model = EncoderDecoderWPointerModel(
            encoder=encoder, decoder=decoder, max_src_len=max_position
        )

        # similar to real data
        # e.g. '[CLS] Directions to Lowell [SEP]'
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2]])
        # e.g. '[IN:GET_DIRECTIONS Directions to [SL:DESTINATION Lowell]]'
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq, decoder_input_ids=tgt_seq, pointer_mask=mask)[0]

        expected_shape = (1, tgt_seq.shape[1], tgt_vocab_size + src_seq.shape[1])
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_shape_on_real_data_batched(self):
        set_seed(42)
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

        model = EncoderDecoderWPointerModel(
            encoder=encoder, decoder=decoder, max_src_len=max_position
        )

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0], [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor(
            [
                [8, 6, 4, 10, 11, 8, 5, 1, 12, 7, 7, 0, 0],
                [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7],
            ]
        )
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq, decoder_input_ids=tgt_seq, pointer_mask=mask)[0]

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

        model = EncoderDecoderWPointerModel(encoder=encoder, decoder=decoder, max_src_len=7)

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0], [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor(
            [
                [8, 6, 4, 10, 11, 8, 5, 1, 12, 7, 7, 0, 0],
                [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7],
            ]
        )
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0]])

        loss = model(
            input_ids=src_seq, decoder_input_ids=tgt_seq, pointer_mask=mask, labels=tgt_seq,
        )[0]

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertGreater(loss, 0)

    def test_save_load(self):
        src_vocab_size = 23
        tgt_vocab_size = 17

        model = EncoderDecoderWPointerModel.from_parameters(
            layers=1,
            hidden=32,
            heads=2,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            max_src_len=7,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )

        input_ids = torch.randint(src_vocab_size, size=(3, 7))
        tgt_sequence = torch.randint(tgt_vocab_size, size=(3, 11))
        decoder_input_ids = tgt_sequence[:, :-1].contiguous()
        labels = tgt_sequence[:, 1:].contiguous()

        expected_output = model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels
        )

        os.mkdir(self.output_dir)
        model.save_pretrained(self.output_dir)

        loaded_model = EncoderDecoderWPointerModel.from_pretrained(self.output_dir)
        self.assertDictEqual(model.config.to_dict(), loaded_model.config.to_dict())
        for i, (p1, p2) in enumerate(zip(model.parameters(), loaded_model.parameters())):
            self.assertTrue(torch.allclose(p1, p2))

        output = loaded_model(
            input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels
        )

        self.assertEqual(len(output), len(expected_output))
        self.assertTrue(torch.allclose(expected_output[0], output[0]))  # loss
        self.assertTrue(torch.allclose(expected_output[1], output[1]))  # logits


class EncoderDecoderWPointerOverfitTest(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.output_dir = next(tempfile._get_candidate_names())

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        if os.path.exists("runs"):
            shutil.rmtree("runs")

    def _prepare_data(self, model):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)

        vocab = {
            "[",
            "]",
            "IN:",
            "SL:",
            "GET_DIRECTIONS",
            "DESTINATION",
            "DATE_TIME_DEPARTURE",
            "GET_ESTIMATED_ARRIVAL",
        }
        schema_tokenizer = TopSchemaTokenizer(vocab, tokenizer)

        source_texts = [
            "Directions to Lowell",
            "Get directions to Mountain View",
        ]
        schema_texts = [
            "[IN:GET_DIRECTIONS Directions to [SL:DESTINATION Lowell]]",
            "[IN:GET_DIRECTIONS Get directions to [SL:DESTINATION Mountain View]]",
        ]

        pairs = [schema_tokenizer.encode_pair(t, s) for t, s in zip(schema_texts, source_texts)]

        dataset = PointerDataset.from_pair_items(pairs)
        dataset.torchify()

        return dataset, tokenizer, schema_tokenizer

    def _train_model(self, model, dataset, lr, epochs):
        train_args = transformers.TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            num_train_epochs=epochs,
            seed=42,
            learning_rate=lr,
        )
        transformers.trainer.is_wandb_available = lambda: False  # workaround to turn off wandb

        meter = MetricsMeter(stop_token_ids=[])

        trainer = Seq2SeqTrainer(
            model,
            train_args,
            train_dataset=dataset,
            data_collator=Seq2SeqDataCollator(
                model.encoder.embeddings.word_embeddings.padding_idx
            ).collate_batch,
            eval_dataset=dataset,
            compute_metrics=meter.compute_metrics,
        )
        # a trick to reduce the amount of logging
        trainer.is_local_master = lambda: False

        train_out = trainer.train()
        eval_out = trainer.evaluate()

        return model, train_out, eval_out

    def test_overfit(self):
        # NOTE: slow test

        for model in MODELS:
            with self.subTest(model):
                dataset, tokenizer, schema_tokenizer = self._prepare_data(model)

                src_maxlen, _ = dataset.get_max_len()

                model = EncoderDecoderWPointerModel.from_parameters(
                    layers=3,
                    hidden=128,
                    heads=2,
                    max_src_len=src_maxlen,
                    src_vocab_size=tokenizer.vocab_size,
                    tgt_vocab_size=schema_tokenizer.vocab_size,
                )

                model, train_out, eval_out = self._train_model(model, dataset, 1e-3, 30)
                pprint("Training output")
                pprint(train_out)
                pprint("Evaluation output")
                pprint(eval_out)

                # accuracy should be 1.0 and eval loss should be around 0.9
                self.assertGreater(eval_out["eval_accuracy"], 0.99)

    def test_overfit_bert(self):
        # NOTE: very slow test

        for model in MODELS:
            with self.subTest(model):
                dataset, tokenizer, schema_tokenizer = self._prepare_data(model)

                src_maxlen, _ = dataset.get_max_len()

                encoder = transformers.AutoModel.from_pretrained(model)

                decoder = transformers.BertModel(
                    transformers.BertConfig(
                        is_decoder=True,
                        vocab_size=schema_tokenizer.vocab_size + src_maxlen,
                        hidden_size=encoder.config.hidden_size,
                        intermediate_size=encoder.config.intermediate_size,
                        num_hidden_layers=encoder.config.num_hidden_layers,
                        num_attention_heads=encoder.config.num_attention_heads,
                        pad_token_id=schema_tokenizer.pad_token_id,
                    )
                )

                model = EncoderDecoderWPointerModel(
                    encoder=encoder, decoder=decoder, max_src_len=src_maxlen
                )

                model, train_out, eval_out = self._train_model(model, dataset, 1e-4, 50)

                pprint("Training output")
                pprint(train_out)
                pprint("Evaluation output")
                pprint(eval_out)

                # accuracy should be 1.0 and eval loss should be around 0.9
                self.assertGreater(eval_out["eval_accuracy"], 0.99)

    def test_overfit_generate(self):
        # NOTE: slow test

        for model in MODELS:
            with self.subTest(model):
                dataset, tokenizer, schema_tokenizer = self._prepare_data(model)

                src_maxlen, _ = dataset.get_max_len()

                model = EncoderDecoderWPointerModel.from_parameters(
                    layers=3,
                    hidden=128,
                    heads=2,
                    max_src_len=src_maxlen,
                    src_vocab_size=tokenizer.vocab_size,
                    tgt_vocab_size=schema_tokenizer.vocab_size,
                )

                model, _, _ = self._train_model(model, dataset, 1e-3, 30)

                example = dataset[0]
                input_ids = example.input_ids.unsqueeze(0)
                labels = example.labels.unsqueeze(0)
                pointer_mask = example.pointer_mask.unsqueeze(0)
                max_len = len(example.decoder_input_ids)

                for num_beams in (1, 4):
                    with self.subTest(msg=f"num_beams={num_beams} (1 is greedy)"):
                        generated = model.generate(
                            input_ids=input_ids,
                            pointer_mask=pointer_mask,
                            max_length=max_len,
                            num_beams=num_beams,
                            pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=schema_tokenizer.bos_token_id,
                            eos_token_id=schema_tokenizer.eos_token_id,
                        )

                        # trim EOS for expected
                        self.assertTrue(torch.all(generated[0] == labels[0][:-1]))

                        decoded = [
                            schema_tokenizer.decode(generated[i], input_ids[i])
                            for i in range(len(generated))
                        ]
                        pprint(decoded)

    def test_overfit_generate_batched(self):
        set_seed(42)
        # NOTE: slow test

        for model in MODELS:
            with self.subTest(model):
                dataset, tokenizer, schema_tokenizer = self._prepare_data(model)

                src_maxlen, _ = dataset.get_max_len()

                model = EncoderDecoderWPointerModel.from_parameters(
                    layers=3,
                    hidden=128,
                    heads=2,
                    max_src_len=src_maxlen,
                    src_vocab_size=tokenizer.vocab_size,
                    tgt_vocab_size=schema_tokenizer.vocab_size,
                )

                model, _, _ = self._train_model(model, dataset, 1e-3, 30)

                dl = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=2,
                    collate_fn=Seq2SeqDataCollator(
                        model.encoder.embeddings.word_embeddings.padding_idx
                    ).collate_batch,
                )

                example = next(iter(dl))
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                labels = example["labels"]
                pointer_mask = example["pointer_mask"]
                max_len = max(map(len, example["decoder_input_ids"]))

                for num_beams in (1, 4):
                    with self.subTest(msg=f"num_beams={num_beams} (1 is greedy)"):
                        generated = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=max_len,
                            num_beams=num_beams,
                            pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=schema_tokenizer.bos_token_id,
                            eos_token_id=schema_tokenizer.eos_token_id,
                            pointer_mask=pointer_mask,
                        )

                        for generated_item, expected_item in zip(generated, labels):
                            # trim EOS for expected
                            self.assertTrue(torch.all(generated_item == expected_item[:-1]))

                        decoded = [
                            schema_tokenizer.decode(generated[i], input_ids[i])
                            for i in range(len(generated))
                        ]
                        pprint(decoded)
