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
import unittest

import torch
import numpy as np

from new_semantic_parsing import utils
from new_semantic_parsing.dataclasses import Seq2SeqEvalPrediciton


class TopSchemaGetVocabularyTest(unittest.TestCase):
    def test_get_vocab(self):
        # example from TOP dataset arxiv.org/abs/1810.07942
        schema_str = (
            "[IN:GET_ESTIMATED_ARRIVAL What time will I arrive at "
            "[SL:DESTINATION [IN:GET_LOCATION_HOME [SL:CONTACT_RELATED "
            "my ] [SL:TYPE_RELATION Mom ] 's house ] ] if I leave "
            "[SL:DATE_TIME_DEPARTURE in five minutes ] ? ]"
        )
        schema_voc = {
            "[",
            "]",
            "IN:",
            "SL:",
            "CONTACT_RELATED",
            "DATE_TIME_DEPARTURE",
            "DESTINATION",
            "GET_ESTIMATED_ARRIVAL",
            "GET_LOCATION_HOME",
            "TYPE_RELATION",
        }

        res = utils.get_vocab_top_schema(schema_str)
        self.assertSetEqual(res, schema_voc)


class TestGetModelType(unittest.TestCase):
    def test_model_type(self):
        model_type = utils.get_model_type("distilbert-base-uncased")
        self.assertEqual(model_type, "distilbert")


class TestGetMetrics(unittest.TestCase):
    def test_metrics(self):
        x = [
            np.array([1, 2, 3, 4, 5, 6]),
            np.array([1, 3, 5, 7, 9]),
            np.array([19, 18, 17, 16, 18, 5, 19]),
        ]

        x_logits = []
        for i, x_i in enumerate(x):
            logit = np.zeros([len(x_i), 20])
            for j, x_ij in enumerate(x_i):
                logit[j, x_ij] = 1.0
            x_logits.append(logit)

        y = [
            np.array([3, 2, 8, 4, 5, 50]),
            np.array([1, 3, 5, 7, 9]),
            np.array([19, 18, 17, 16, 18, 5, 1]),
        ]

        # NOTE: we expect micro averaging
        expected_accuracy = 0.777777777777  # not truncated
        expected_exact_match = 0.66666666666  # truncated until 5

        meter = utils.MetricsMeter(stop_token_ids=[5])

        metrics = meter.compute_metrics(Seq2SeqEvalPrediciton(x_logits, y))

        self.assertAlmostEqual(metrics["accuracy"], expected_accuracy)
        self.assertAlmostEqual(metrics["exact_match"], expected_exact_match)

    def test_metrics_with_padding(self):
        x = [
            np.array([1, 2, 3, 4, 5, 6]),
            np.array([1, 3, 5, 7, 9]),
            np.array([19, 18, 17, 16, 18, 13, 19]),
        ]

        x_logits = []
        for i, x_i in enumerate(x):
            logit = np.zeros([len(x_i), 20])
            for j, x_ij in enumerate(x_i):
                logit[j, x_ij] = 1.0
            x_logits.append(logit)

        y = [
            np.array([3, 2, 8, 4, 5, 5]),
            np.array([1, 3, 5, 7, 9]),
            np.array([19, 8, 17, 16, 18, 5, 1]),
        ]
        m = [
            np.array([0, 1, 0, 1, 1, 0]),
            np.array([1, 1, 1, 1, 1]),
            np.array([1, 1, 0, 1, 1, 0, 0]),
        ]

        # NOTE: we expect micro averaging
        expected_accuracy = 0.91666666666
        expected_exact_match = 0.33333333333  # mask is not used for EM, only truncation

        meter = utils.MetricsMeter(stop_token_ids=[5])

        metrics = meter.compute_metrics(Seq2SeqEvalPrediciton(x_logits, y, m))

        self.assertAlmostEqual(metrics["accuracy"], expected_accuracy)
        self.assertAlmostEqual(metrics["exact_match"], expected_exact_match)


class TestGetMetircsTorch(unittest.TestCase):
    def test_metrics_computation(self):
        x = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 0, 0, 0],
                [1, 3, 5, 7, 9, 0, 0, 0, 0],
                [19, 18, 17, 16, 18, 13, 19, 0, 0],
            ]
        )

        y = torch.tensor(
            [
                [3, 2, 8, 4, 5, 5, 0, 0, 0],
                [1, 3, 5, 7, 9, 0, 0, 0, 0],
                [19, 8, 17, 16, 18, 5, 1, 0, 0],
            ]
        )

        m = torch.tensor(
            [
                [0, 1, 0, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 0],
            ]
        )

        assert x.shape == y.shape
        assert y.shape == m.shape

        stop_tokens = [5, 0]
        metrics = utils.compute_metrics_from_batch(x, y, m, stop_tokens)

        # NOTE: we expect micro averaging
        expected_accuracy = 0.91666666666
        expected_exact_match = 0.33333333333  # mask is not used for EM, only truncation

        self.assertTrue(metrics["accuracy"].numpy(), expected_accuracy)
        self.assertTrue(metrics["exact_match"].numpy(), expected_exact_match)
