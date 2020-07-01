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
import unittest
from unittest.mock import patch, MagicMock

from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer


class TransformersTokenizerMock:
    cls_token = '[CLS]'
    cls_token_id = 101

    def encode(self, x, add_special_tokens=False):
        subtokens = x.split(',')
        return [int(t[3:]) for t in subtokens]

    def decode(self, x):
        return ' '.join([f'tok{i}' for i in x])

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def save_pretrained(self, path):
        pass


class TopSchemaTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = next(tempfile._get_candidate_names())

    def tearDown(self):
        if os.path.exists(self.tmpdirname):
            shutil.rmtree(self.tmpdirname)

    def test_tokenize(self):
        """
        Test cases are examples from TOP dataset arxiv.org/abs/1810.07942
        """
        schema_str = '[IN:INTENT1 tok1 tok2 tok3 [SL:SLOT1 tok4 tok5 ] ]'
        schema_tok = '[ IN: INTENT1 tok1 tok2 tok3 [ SL: SLOT1 tok4 tok5 ] ]'.split(' ')

        res = TopSchemaTokenizer.tokenize(schema_str)
        self.assertSequenceEqual(res, schema_tok)

        schema_str = ('[IN:GET_EVENT Any [SL:CATEGORY_EVENT festivals ] '
                      '[SL:DATE_TIME this weekend ] ]')

        schema_tok = ('[ IN: GET_EVENT Any [ SL: CATEGORY_EVENT festivals ] '
                      '[ SL: DATE_TIME this weekend ] ]').split(' ')

        res = TopSchemaTokenizer.tokenize(schema_str)
        self.assertSequenceEqual(res, schema_tok)

        schema_str = ("[IN:GET_ESTIMATED_ARRIVAL What time will I arrive at "
                      "[SL:DESTINATION [IN:GET_LOCATION_HOME [SL:CONTACT_RELATED "
                      "my ] [SL:TYPE_RELATION Mom ] 's house ] ] if I leave "
                      "[SL:DATE_TIME_DEPARTURE in five minutes ] ? ]")

        schema_tok = ("[ IN: GET_ESTIMATED_ARRIVAL What time will I arrive at "
                      "[ SL: DESTINATION [ IN: GET_LOCATION_HOME [ SL: CONTACT_RELATED "
                      "my ] [ SL: TYPE_RELATION Mom ] 's house ] ] if I leave "
                      "[ SL: DATE_TIME_DEPARTURE in five minutes ] ? ]")
        schema_tok = schema_tok.split(' ')

        tokens = TopSchemaTokenizer.tokenize(schema_str)
        self.assertSequenceEqual(tokens, schema_tok)

    def test_encode_nocls(self):
        vocab = {'[', ']', 'IN:', 'INTENT1', 'SL:', 'SLOT1'}
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer)

        schema_str = '[IN:INTENT1 tok6,tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]'
        source_tokens = [6, 2, 31, 42, 5]
        # note that TransformersTokenizerMock splits tok6,tok2 into two subtokens
        # note that the vocabulary is sorted
        expected_ids = [tokenizer.bos_token_id, 7, 3, 4, 9, 10, 11, 7, 5, 6, 12, 13, 8, 8, tokenizer.eos_token_id]

        ids = tokenizer.encode(schema_str, source_tokens)
        self.assertSequenceEqual(ids, expected_ids)

    def test_encode_cls(self):
        vocab = ['[', ']', 'IN:', 'INTENT1', 'SL:', 'SLOT1']
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer)

        schema_str = '[IN:INTENT1 tok6,tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]'
        source_tokens = [TransformersTokenizerMock.cls_token_id, 6, 2, 31, 42, 5]
        # note that TransformersTokenizerMock splits tok6,tok2 into two subtokens
        expected_ids = [tokenizer.bos_token_id, 7, 3, 4, 10, 11, 12, 7, 5, 6, 13, 14, 8, 8, tokenizer.eos_token_id]

        ids = tokenizer.encode(schema_str, source_tokens)
        self.assertSequenceEqual(ids, expected_ids)

    def test_keywords_in_text(self):
        vocab = ['[', ']', 'IN:', 'INTENT1', 'SL:', 'SLOT1', 'SLT1']
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer)

        # i.e. SLOT1 after tok2 is just a token which is written exactly like a schema word
        schema_str = '[IN:INTENT1 tok6 tok2 SLT1 tok31 [SL:SLOT1 tok42 tok5 ] ]'
        source_tokens = [6, 2, 1, 31, 42, 5]
        expected_ids = [tokenizer.bos_token_id, 8, 3, 4, 10, 11, 12, 13, 8, 5, 6, 14, 15, 9, 9, tokenizer.eos_token_id]

        ids = tokenizer.encode(schema_str, source_tokens)
        self.assertSequenceEqual(ids, expected_ids)

    def test_save_load(self):
        vocab = ['[', ']', 'IN:', 'INTENT1', 'SL:', 'SLOT1']
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer)

        schema_str = '[IN:INTENT1 tok6,tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]'
        source_tokens = [6, 2, 31, 42, 5]

        ids = tokenizer.encode(schema_str, source_tokens)

        tokenizer.save(self.tmpdirname, encoder_model_type='test_type')

        with patch('new_semantic_parsing.schema_tokenizer.transformers.AutoTokenizer.from_pretrained',
                   MagicMock(return_value=TransformersTokenizerMock())):
            loaded_tokenizer = TopSchemaTokenizer.load(self.tmpdirname)
        self.assertSetEqual(set(loaded_tokenizer._vocab), set(tokenizer._vocab))

        new_ids = loaded_tokenizer.encode(schema_str, source_tokens)
        self.assertSequenceEqual(ids, new_ids)

    def test_decode(self):
        vocab = {'[', ']', 'IN:', 'INTENT1', 'SL:', 'SLOT1'}
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer)

        schema_str = '[IN:INTENT1 tok6 tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]'
        source_tokens = [6, 2, 31, 42, 5]
        # note that TransformersTokenizerMock splits tok6,tok2 into two subtokens
        # note that the vocabulary is sorted
        expected_ids = [7, 3, 4, 9, 10, 11, 7, 5, 6, 12, 13, 8, 8]

        schema_decoded = tokenizer.decode(expected_ids, source_tokens)

        self.assertEqual(schema_str, schema_decoded)

    def test_postprocess_punct(self):
        text = "[What is this?]"
        expected = "[What is this ?]"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)

        text = "[This is nothing ! ]"
        expected ="[This is nothing ! ]"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)

        text = "7;45"
        expected ="7 ; 45"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)

    def test_postprocess_apostrophe(self):
        text = "[What's]"
        expected = "[What 's]"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)

        text = "[I didn't do this.]"
        expected = "[I didn't do this .]"
        postprocessed = TopSchemaTokenizer.postprocess(text)

        self.assertSequenceEqual(expected, postprocessed)

        text = "[[Your ] ' s]"
        expected = "[[Your ] 's]"
        postprocessed = TopSchemaTokenizer.postprocess(text)
        self.assertSequenceEqual(expected, postprocessed)
