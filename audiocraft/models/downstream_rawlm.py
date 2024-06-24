# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import torch
from torch import nn

from ..modules.conditioners import (ConditionFuser, ConditioningAttributes,
                                    ConditioningProvider)
from .lm import ConditionTensors
from .rawlm import RawLMModel

logger = logging.getLogger(__name__)


class DownstreamRawLMModel(RawLMModel):
    def __init__(
        self,
        condition_provider: ConditioningProvider,
        fuser: ConditionFuser,
        max_tr: float = 256.0,
        tr_precision: float = 0.2,
        space_dim: int = 1024,
        num_classes: int = 20,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_first: bool = False,
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
            condition_provider=condition_provider,
            fuser=fuser,
            max_tr=max_tr,
            tr_precision=tr_precision,
            space_dim=space_dim,
            dim=dim,
            num_heads=num_heads,
            hidden_scale=hidden_scale,
            norm=norm,
            norm_first=norm_first,
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
        self.cls_emb = nn.Parameter(torch.randn(self.space_dim, 1))
        self.num_class_linear = nn.Linear(self.space_dim, num_classes)

    def add_cls_emb(
        self,
        bold: torch.Tensor,
        tr_tensor: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        B, _, _ = bold.shape
        cls_embedding_expanded = self.cls_emb.unsqueeze(0).repeat(B, 1, 1)
        cls_tr_tensor = torch.zeros(B, 1).long().to(bold.device) + self.tr_emb_pad_id

        return torch.cat((bold, cls_embedding_expanded), dim=-1), torch.cat((tr_tensor, cls_tr_tensor), dim=-1)

    def compute_predictions(
        self,
        bold: torch.Tensor,  # [B, dim, T]
        bold_durations: tp.List[int],  # [B,]
        tr: tp.List[float],  # [B, ]
        conditions: tp.List[ConditioningAttributes],
        condition_tensors: tp.Optional[ConditionTensors] = None,
    ) -> torch.Tensor:
        model = self if self._fsdp is None else self._fsdp
        bold, bold_durations = self.add_start_emb(bold, bold_durations)
        tr_tensor = self.tr_preprocess(tr, bold.shape[-1], bold_durations).to(bold.device)

        bold, tr_tensor = self.add_cls_emb(bold, tr_tensor)

        output = model(bold, tr_tensor, conditions, condition_tensors)
        return self.num_class_linear(output[:, -1, :])  # output only last time
