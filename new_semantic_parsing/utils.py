# Copyright 2020 Google LLC
# Copyright 2020 The HuggingFace Inc. team.
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
"""Utilities used across code.

Include fixing random seeds, metrics computation, learning rate selection, model loading, and prediction.
"""

import random
import torch
import numpy as np
import transformers

from tqdm.auto import tqdm

from new_semantic_parsing.dataclasses import Seq2SeqEvalPrediciton


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_vocab_top_schema(text):
    schema_tokens = {"[", "]", "IN:", "SL:"}

    text = text.replace("[", "")
    text = text.replace("]", "")

    for token in text.split(" "):
        if token[:3] in ["IN:", "SL:"]:
            schema_tokens.add(token[3:])
    return schema_tokens


def get_src_pointer_mask(input_ids, tokenizer: transformers.PreTrainedTokenizer):
    """Compute mask which ignores padding and special tokens"""
    mask = np.ones(len(input_ids))
    if input_ids[0] == tokenizer.cls_token_id:
        mask[0] = 0
    for i, token_id in enumerate(input_ids):
        if token_id in (tokenizer.sep_token_id, tokenizer.pad_token_id):
            mask[i] = 0
    return mask


class MetricsMeter:
    def __init__(self, stop_token_ids):
        """
        :param stop_tokens: tokens after which decoding stops (typically EOS and PAD)
        """
        self.stop_token_ids = stop_token_ids

    def _truncate(self, pred):
        i = 0
        for i, idx in enumerate(pred):
            if idx in self.stop_token_ids:
                break
        return pred[:i]

    def compute_metrics(self, eval_prediction: Seq2SeqEvalPrediciton):
        """
        :param eval_prediction: each field of the dataclass corresponds to a list,
            list element is a single example
        :return:
            dict with keys
                `accuracy` - fraction of correctly predicted tokens under the mask
                `exact_match` - fraction of correctly predicted sequences (sequences are truncated after EOS)
        """
        if len(eval_prediction.predictions[0].shape) != 2:
            raise ValueError(
                "eval_prediction.predictions should be a list of predictions, "
                "expected prediction shape to be (seq_len, vocab_dim), got "
                f"{eval_prediction.predictions[0].shape} instead"
            )

        predictions = [np.argmax(p, axis=-1) for p in eval_prediction.predictions]
        labels = eval_prediction.label_ids

        # accuracy is computed under a known mask
        # a bit more precisionish behavior, than just accuracy (can be whatather outside the mask)
        label_masks = eval_prediction.label_masks or [np.ones_like(p) for p in predictions]

        total_tokens = sum(np.sum(m) for m in label_masks)
        # (p == l) & (1 ^ m) <=> correct tokens which are not masked (masked tokens have mask == 0)
        accuracy = sum(np.sum((p == l) & m) for p, l, m in zip(predictions, labels, label_masks))
        accuracy /= total_tokens

        # trauncate until EOS token
        # for exact match we consider all tokens until EOS/PAD
        # this is closer to inference setup when generation stops after EOS/PAD
        truncated_predictions = [self._truncate(p) for p in predictions]
        truncated_labels = [self._truncate(l) for l in labels]

        exact_match = sum(np.all(p == l) for p, l in zip(truncated_predictions, truncated_labels))
        exact_match /= len(truncated_predictions)

        return {
            "accuracy": accuracy,
            "exact_match": exact_match,
        }


def compute_metrics_from_batch(predictions, labels, masks, stop_tokens):
    """Compute metrics where all predictions, labels and masks are torch.tensor"""
    device = predictions.device

    # correct tokens which are not masked (masked tokens have mask == 0)
    n_correct = ((predictions == labels) & masks.bool()).sum()
    n_total = masks.sum()
    accuracy = n_correct / n_total.float()

    # trauncate until EOS token
    # for exact match we consider all tokens until EOS/PAD
    # this is closer to inference setup when generation stops after EOS/PAD

    def truncate(pred):
        i = 0
        for i, idx in enumerate(pred):
            if idx in stop_tokens:
                break
        return pred[:i]

    truncated_preds = [truncate(p) for p in predictions.detach().unbind()]
    truncated_labels = [truncate(l) for l in labels.detach().unbind()]

    exact_match = sum(
        (p.shape == l.shape and torch.all(p == l))
        for p, l in zip(truncated_preds, truncated_labels)
    )
    exact_match = torch.tensor(exact_match, device=device)
    exact_match = exact_match.float() / len(truncated_preds)

    return {"accuracy": accuracy, "exact_match": exact_match}


def get_lr(model: transformers.PreTrainedModel):
    """Get optimal learning rate according to the Scaling Laws.

    https://arxiv.org/abs/2001.08361
    lr ~= 0.003239 - 0.0001395 log(n_non_embedding_params)
    """

    if hasattr(model, "embeddings"):
        n_embedding_params = get_n_embed_params_trainable(model.embeddings)
    elif hasattr(model, "encoder") and hasattr(model, "decoder"):
        n_embed_encoder = get_n_embed_params_trainable(model.encoder.embeddings)
        n_embed_decoder = get_n_embed_params_trainable(model.decoder.embeddings)
        n_embedding_params = n_embed_encoder + n_embed_decoder
    else:
        raise ValueError(
            "Model object should have .embeddings or.encoder.embeddings and .decoder.embeddings"
        )

    n_non_embedding_params = model.num_parameters(only_trainable=True) - n_embedding_params
    return 0.003239 - 0.0001395 * np.log(n_non_embedding_params)


def get_n_embed_params_trainable(embeddings):
    """
    :param embeddings: BertEmbeddings
    :returns: number of trainable embedding parameters
    """
    n_params = 0

    tok = embeddings.word_embeddings
    if tok.training:
        n_params += tok.num_embeddings * tok.embedding_dim

    pos = embeddings.position_embeddings
    if pos.training:
        n_params += pos.num_embeddings * pos.embedding_dim

    typ = embeddings.token_type_embeddings
    if typ.training:
        n_params += typ.num_embeddings * typ.embedding_dim

    return n_params


def get_model_type(model_name):
    """Search for a largest substring from transformers.CONFIG_MAPPING"""
    candidate = ""

    for name in transformers.CONFIG_MAPPING:
        if name in model_name and len(name) > len(candidate):
            candidate = name

    if len(candidate) == 0:
        raise ValueError(f"{model_name} is not found in transformers.CONFIG_MAPPING")

    return candidate


def iterative_prediction(model, dataloader, schema_tokenizer, max_len, num_beams, device="cpu"):
    predictions_ids = []
    predictions_str = []
    text_tokenizer = schema_tokenizer.src_tokenizer

    for batch in tqdm(dataloader, desc="generation"):
        prediction_batch: torch.LongTensor = model.generate(
            input_ids=batch["input_ids"].to(device),
            pointer_mask=batch["pointer_mask"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_length=max_len,
            num_beams=num_beams,
            pad_token_id=text_tokenizer.pad_token_id,
            bos_token_id=schema_tokenizer.bos_token_id,
            eos_token_id=schema_tokenizer.eos_token_id,
        )

        for i, prediction in enumerate(prediction_batch):
            prediction = [
                p for p in prediction.cpu().numpy() if p not in schema_tokenizer.special_ids
            ]
            predictions_ids.append(prediction)

            prediction_str: str = schema_tokenizer.decode(
                prediction, batch["input_ids"][i], skip_special_tokens=True,
            )
            predictions_str.append(prediction_str)

    return predictions_ids, predictions_str
