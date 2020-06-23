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
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers


class EncoderDecoderWPointerModel(transformers.EncoderDecoderModel):
    """
    Encoder-decoder with pointer model as in arxiv.org/abs/2001.11458
    """
    def __init__(self, encoder, decoder, maximal_pointer, model_args=None, **kwargs):
        """
        :param encoder: transformers.PreTrainedModel
        :param decoder: transformers.BertModel, BertFor*Model are not supported
        :param model_args: argparse arguments, architecture parameters
        :param maximal_pointer: maximum length of the encoder sequence, defines the maximum possible pointer index
        """
        super().__init__(encoder=encoder, decoder=decoder, **kwargs)
        self.model_args = model_args

        # unwrap all model_args here for clarity
        decoder_head_type = getattr(model_args, 'decoder_head_type', 'ffn')
        use_pointer_bias = getattr(model_args, 'use_pointer_bias', False)

        self.enc_dec_proj = None
        if encoder.config.hidden_size != decoder.config.hidden_size:
            self.enc_dec_proj = nn.Linear(encoder.config.hidden_size, decoder.config.hidden_size)

        # this gives a schema vocab size (without pointer embeddings)
        output_vocab_size = decoder.config.vocab_size - maximal_pointer
        if decoder_head_type == 'ffn':
            head_config = deepcopy(decoder.config)
            head_config.vocab_size = output_vocab_size

            # Linear -> activation -> LayerNorm -> Linear
            # from config only .hidden_size, .hidden_act, .layer_norm_eps and .vocab_size are used
            self.lm_head = transformers.modeling_bert.BertLMPredictionHead(head_config)
        elif model_args.decoder_head_type == 'linear':
            self.lm_head = nn.Linear(decoder.config.hidden_size, output_vocab_size)
        else:
            raise ValueError(model_args.decoder_head_type)

        # One does not simply ties weights of embeddings with different vocabularies
        # # lm_head.decoder is just a linear layer
        # self._tie_or_clone_weights(self.lm_head.decoder,
        #                            self.decoder.get_input_embeddings())

        self.decoder_q_proj = nn.Linear(self.decoder.config.hidden_size,
                                        self.decoder.config.hidden_size,
                                        bias=use_pointer_bias)

    @classmethod
    def from_parameters(
        cls,
        layers,
        hidden,
        heads,
        src_vocab_size,
        tgt_vocab_size,
        maximal_pointer,
        decoder_layers=None,
        decoder_hidden=None,
        decoder_heads=None,
        encoder_pad_token_id=0,
        decoder_pad_token_id=None,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        model_args=None,
    ):
        """
        :param layers: number of layers for encoder and for decoder
        :param hidden: hidden size
        :param heads: number of attention heads
        :param src_vocab_size: source vocabulary size
        :param tgt_vocab_size: size of the target vocabulary (excluding pointers)
        :param encoder_pad_token_id: pad id to ignore in the encoder input
        :param decoder_pad_token_id: pad id to ignore in the decoder input, equal to encoder_pad_token_id by default
        :return: EncoderDecoderWPointerModel
        """
        encoder_config = transformers.BertConfig(
            hidden_size=hidden,
            intermediate_size=4 * hidden,
            vocab_size=src_vocab_size,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            pad_token_id=encoder_pad_token_id,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_pad_token_id = decoder_pad_token_id or encoder_pad_token_id
        decoder_hidden = decoder_hidden or hidden
        decoder_layers = decoder_layers or layers
        decoder_heads = decoder_heads or heads

        decoder_config = transformers.BertConfig(
            hidden_size=decoder_hidden,
            intermediate_size=4 * decoder_hidden,
            vocab_size=tgt_vocab_size + maximal_pointer,
            is_decoder=True,
            num_hidden_layers=decoder_layers,
            num_attention_heads=decoder_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            pad_token_id=decoder_pad_token_id,
        )
        decoder = transformers.BertModel(decoder_config)

        return cls(encoder, decoder, maximal_pointer=maximal_pointer, model_args=model_args)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        decoder_inputs_embeds=None,
        pointer_mask=None,
        decoder_pointer_mask=None,
        labels=None,
        **kwargs,
    ):
        """
        Note that all masks are FloatTensors and equal 1 for keep and 0 for mask

        :param input_ids: LongTensor of shape (batch_size, src_seq_len)
        :param inputs_embeds: alternative to input_ids, shape (batch_size, src_seq_len, embed_dim)
        :param attention_mask: FloatTensor of shape (batch_size, src_seq_len), padding mask for the encoder
        :param head_mask: FloatTensor of shape (num_heads,) or (num_layers, num_heads)
        :param encoder_outputs: alternative to input_ids, tuple(last_hidden_state, hidden_states, attentions)
            last_hidden_state has shape (batch_size, src_seq_len, hidden)
        :param decoder_input_ids: LongTensor of shape (batch_size, tgt_seq_len)
        :param decoder_inputs_embeds: alternative to decoder_input_ids, shape (batch_size, tgt_seq_len, embed_dim
        :param decoder_attention_mask: FloatTensor of shape (batch_size, tgt_seq_lqn), padding mask for the encoder
            decoder input padding mask, causal mask is generated inside the decoder class
        :param decoder_head_mask: FloatTensor of shape (num_heads,) or (num_layers, num_heads)
        :param pointer_mask: FloatTensor of shape (batch_size, src_seq_len), padding mask for the pointer
        :param labels: LongTensor of shape (batch_size, tgt_seq_len), typically equal decoder_input_ids
        :param kwargs:
        :return:
            tuple of
                loss (if labels are specified)
                combined_logits tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size + src_vocab_size),
                decoder outputs,
                encoder outputs,
        """
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs[0]

        if self.enc_dec_proj is not None:
            encoder_hidden_states = self.enc_dec_proj(encoder_hidden_states)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            **kwargs_decoder,
        )

        decoder_hidden_states = decoder_outputs[0]

        # compute pointer scores via attending from decoder hiddens to encoder hiddens

        query = self.decoder_q_proj(decoder_hidden_states)  # (bs, tgt_len, decoder_hidden)
        keys = encoder_hidden_states  # (bs, src_len, encoder_hidden)

        # NOTE: this implementaion is computationally inefficient during inference
        attention_scores = query @ keys.transpose(1, 2)  # (bs, tgt_len, src_len)

        # mask becomes 0 for all 1 (keep) positions and -1e4 in all 0 (mask) positions
        # NOTE: we can use this mask to additionaly guide the model
        pointer_mask = self._get_pointer_attention_mask(
            pointer_mask, attention_scores.shape,
        )
        attention_scores = attention_scores + pointer_mask
        # attention_scores = attention_scores * attention_scores.shape[-1] ** -0.5

        # NOTE: maybe add some kind of normalization between dec_logits?
        decoder_logits = self.lm_head(decoder_hidden_states)  # (bs, tgt_len, tgt_vocab_size)
        combined_logits = torch.cat([decoder_logits, attention_scores], dim=-1)

        if labels is None:
            return (combined_logits,) + decoder_outputs + encoder_outputs

        loss = self._compute_loss(combined_logits, labels)
        return (loss, combined_logits) + decoder_outputs + encoder_outputs

    def _compute_loss(self, input, target):
        return F.cross_entropy(
            input.view(-1, input.shape[-1]),
            target.view(-1),
            ignore_index=self.decoder.embeddings.word_embeddings.padding_idx
        )

    def _get_pointer_attention_mask(self, pointer_attention_mask=None, shape=None, device=None, dtype=None):
        """
        :param pointer_attention_mask: FloatTensor of shape (batch_size, src_seq_len), padding mask for the pointer
            0 for masking and 1 for no masking
        :param shape: alternative to pointer_attention_mask, tuple (batch_size, src_seq_len, tgt_seq_len)
        :param device: torch.device
        :return: FloatTensor of shape (batch_size, 1, src_seq_len)
            attention mask which equals -1e4 for src padding and special tokens and zero otherwise
        """
        device = device or self.device
        dtype = dtype or self.dtype

        if pointer_attention_mask is None:
            bs, _, src_len = shape
            return torch.zeros([bs, 1, src_len], device=device, dtype=dtype)

        # We use -1e4 for masking analogous to Transformers library
        # ideally, this number should depend on dtype and should be
        # bigger for float32 and smaller for float16 and bfloat16
        return ((1. - pointer_attention_mask) * -1e4).unsqueeze(1)
