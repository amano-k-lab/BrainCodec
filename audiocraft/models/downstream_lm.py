# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import torch
from torch import nn

from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.conditioners import (ConditionFuser, ConditioningAttributes,
                                    ConditioningProvider)
from .lm import ConditionTensors, LMModelfMRI

logger = logging.getLogger(__name__)


class DownstreamLMModel(LMModelfMRI):
    def __init__(
        self,
        pattern_provider: CodebooksPatternProvider,
        condition_provider: ConditioningProvider,
        fuser: ConditionFuser,
        max_tr: float = 256.0,
        tr_precision: float = 0.2,
        num_classes: int = 20,
        n_q: int = 8,
        card: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_first: bool = False,
        emb_lr: tp.Optional[float] = None,
        bias_proj: bool = True,
        weight_init: tp.Optional[str] = None,
        depthwise_init: tp.Optional[str] = None,
        zero_bias_init: bool = False,
        cfg_dropout: float = 0,
        cfg_coef: float = 1.0,
        attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {},
        two_step_cfg: bool = False,
        **kwargs
    ):
        super().__init__(
            pattern_provider=pattern_provider,
            condition_provider=condition_provider,
            fuser=fuser,
            max_tr=max_tr,
            tr_precision=tr_precision,
            n_q=n_q,
            card=card,
            dim=dim,
            num_heads=num_heads,
            hidden_scale=hidden_scale,
            norm=norm,
            norm_first=norm_first,
            emb_lr=emb_lr,
            bias_proj=bias_proj,
            weight_init=weight_init,
            depthwise_init=depthwise_init,
            zero_bias_init=zero_bias_init,
            cfg_dropout=cfg_dropout,
            cfg_coef=cfg_coef,
            attribute_dropout=attribute_dropout,
            two_step_cfg=two_step_cfg,
            **kwargs
        )
        self.cls_emb = nn.Parameter(torch.randn(1, self.dim))
        self.num_class_linear = nn.Linear(self.dim, num_classes)

    def forward(
        self,
        sequence: torch.Tensor,
        tr_tensor: torch.Tensor,
        conditions: tp.List[ConditioningAttributes],
        condition_tensors: tp.Optional[ConditionTensors] = None,
    ) -> torch.Tensor:  # noqa: F821
        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, K, S] with K the number of codebooks and
        S the sequence steps, return the logits with shape [B, card, K, S].

        Args:
            indices (torch.Tensor): Indices of the codes to model.
            conditions (list of ConditioningAttributes): Conditions to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            torch.Tensor: Logits.
        """
        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])

        # add cls embedding
        cls_embedding_expanded = self.cls_emb.unsqueeze(0).repeat(B, 1, 1)
        input_ = torch.cat((input_, cls_embedding_expanded), dim=1)

        if condition_tensors is None:
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
            # apply dropout modules
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            # encode conditions and fuse, both have a streaming cache to not recompute when generating.
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, "Shouldn't pass both conditions and condition_tensors."

        input_, cross_attention_input = self.fuser(input_, condition_tensors)

        tr_emb = self.tr_emb(tr_tensor)
        out = self.transformer(input_ + tr_emb, cross_attention_src=cross_attention_input)
        if self.out_norm:
            out = self.out_norm(out)
        return self.num_class_linear(out[:, -1])

    def compute_predictions_for_fmri(
        self,
        bold_codes: torch.Tensor,  # [B, dim, T]
        bold_codes_durations: tp.List[int],  # [B,]
        tr: tp.List[float],  # [B, ]
        conditions: tp.List[ConditioningAttributes],
        condition_tensors: tp.Optional[ConditionTensors] = None,
    ) -> torch.Tensor:
        B, K, T = bold_codes.shape
        bold_codes = bold_codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        (
            sequence_codes,
            sequence_indexes,
            sequence_mask,
        ) = pattern.build_pattern_sequence(bold_codes, self.special_token_id, keep_only_valid_steps=True)
        tr_tensor = self.tr_preprocess(tr, sequence_codes.shape[-1], bold_codes_durations, for_lm=True).to(
            bold_codes.device
        )
        # add pad token for cls embedding
        tr_tensor = torch.cat(
            (
                tr_tensor,
                torch.zeros(B, 1).long().to(bold_codes.device) + self.tr_emb_pad_id,
            ),
            dim=-1,
        )

        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, tr_tensor, conditions, condition_tensors)  # [B, K, S, card]
        return logits
