# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from functools import partial

import torch
from torch import nn

from ..modules.activations import get_activation_fn
from ..modules.conditioners import (AttributeDropout,
                                    ClassifierFreeGuidanceDropout,
                                    ConditionFuser, ConditioningAttributes,
                                    ConditioningProvider)
from ..modules.streaming import State, StreamingModule
from ..modules.transformer import StreamingTransformer, create_norm_fn
from .lm import CFGConditions, ConditionTensors, init_layer

logger = logging.getLogger(__name__)


class RawLMModel(StreamingModule):
    def __init__(
        self,
        condition_provider: ConditioningProvider,
        fuser: ConditionFuser,
        max_tr: float = 256.0,
        tr_precision: float = 0.2,
        space_dim: int = 1024,
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
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.space_dim = space_dim
        self.dim = dim
        self.two_step_cfg = two_step_cfg
        if "activation" in kwargs:
            kwargs["activation"] = get_activation_fn(kwargs["activation"])

        self.tr_precision = tr_precision
        self.tr_emb_pad_id = int(max_tr / tr_precision) + 1

        self.input_linear = nn.Linear(space_dim, dim, bias=bias_proj)
        self.tr_emb = nn.Embedding(self.tr_emb_pad_id + 1, dim, padding_idx=self.tr_emb_pad_id)

        self.start_emb = nn.Parameter(torch.randn(self.space_dim, 1))

        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            norm_first=norm_first,
            **kwargs
        )
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)
        self.output_linear = nn.Linear(dim, space_dim, bias=bias_proj)

        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__["_fsdp"] = None

    def _init_weights(
        self,
        weight_init: tp.Optional[str],
        depthwise_init: tp.Optional[str],
        zero_bias_init: bool,
    ):
        """Initialization of the transformer module weights.

        Args:
            weight_init (str, optional): Weight initialization strategy. See ``get_init_fn`` for valid options.
            depthwise_init (str, optional): Depthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initialize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ["current", "global"]
        assert (
            depthwise_init is None or weight_init is not None
        ), "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert (
            not zero_bias_init or weight_init is not None
        ), "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        init_layer(
            self.tr_emb,
            method=weight_init,
            init_depth=None,
            zero_bias_init=zero_bias_init,
        )

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == "current":
                depth = layer_idx + 1
            elif depthwise_init == "global":
                depth = len(self.transformer.layers)
            init_fn = partial(
                init_layer,
                method=weight_init,
                init_depth=depth,
                zero_bias_init=zero_bias_init,
            )
            tr_layer.apply(init_fn)

        for linear in [self.input_linear, self.output_linear]:
            init_layer(
                linear,
                method=weight_init,
                init_depth=None,
                zero_bias_init=zero_bias_init,
            )

    def tr_preprocess(
        self,
        raw_tr: tp.List[float],
        max_durtion: int,
        bold_durations: tp.List[int],
    ) -> torch.Tensor:
        assert len(raw_tr) == len(bold_durations)
        tr = torch.zeros(len(raw_tr), max_durtion).long()
        for idx, (_tr, bold_d) in enumerate(zip(raw_tr, bold_durations)):
            _tr_tensor = torch.arange(0, bold_d) * _tr
            _tr_tensor = (_tr_tensor / self.tr_precision).floor()
            if bold_d > max_durtion:
                _tr_tensor = _tr_tensor[:max_durtion]
            else:
                pad_length = max_durtion - bold_d
                _tr_tensor = torch.cat((_tr_tensor, self.tr_emb_pad_id * torch.ones(pad_length)))
            tr[idx] = _tr_tensor

        assert (tr <= self.tr_emb_pad_id).all()
        return tr

    def add_start_emb(
        self,
        bold: torch.Tensor,
        bold_durations: tp.Optional[tp.List[int]] = None,
    ) -> tp.Tuple[torch.Tensor, tp.Optional[tp.List[int]]]:
        B, _, _ = bold.shape
        start_embedding_expanded = self.start_emb.unsqueeze(0).repeat(B, 1, 1)
        if bold_durations is not None:
            return torch.cat((start_embedding_expanded, bold), dim=2), [b + 1 for b in bold_durations]
        else:
            return torch.cat((start_embedding_expanded, bold), dim=2), None

    def forward(
        self,
        bold: torch.Tensor,  # [B, dim, T]
        tr_tensor: torch.Tensor,  # [B, T]
        conditions: tp.List[ConditioningAttributes],
        condition_tensors: tp.Optional[ConditionTensors] = None,
    ) -> torch.Tensor:
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

        bold, cross_attention_input = self.fuser(bold, condition_tensors)
        bold = self.input_linear(bold.transpose(1, 2))
        tr_emb = self.tr_emb(tr_tensor)
        out = self.transformer(bold + tr_emb, cross_attention_src=cross_attention_input)
        if self.out_norm:
            out = self.out_norm(out)

        # remove the prefix from the model outputs
        if len(self.fuser.fuse2cond["prepend"]) > 0:
            raise ValueError("prepend is not supported for rawlm, because we add Tr")

        return self.output_linear(out)

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
        output = model(bold, tr_tensor, conditions, condition_tensors)
        return output

    def _sample_next_token(
        self,
        sequence: torch.Tensor,
        tr_seq: torch.Tensor,
        cfg_conditions: CFGConditions,
        unconditional_state: State,
        use_sampling: bool = False,
        cfg_coef: tp.Optional[float] = None,
    ) -> torch.Tensor:
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        if self.two_step_cfg and cfg_conditions != {}:
            assert isinstance(cfg_conditions, tuple), type(cfg_conditions)
            condition_tensors, null_condition_tensors = cfg_conditions
            cond_output = model(sequence, tr_seq, conditions=[], condition_tensors=condition_tensors)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_output = model(
                sequence,
                tr_seq,
                conditions=[],
                condition_tensors=null_condition_tensors,
            )
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            output = uncond_output + (cond_output - uncond_output) * self.cfg_coef
        else:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                # Preparing for CFG, predicting both conditional and unconditional logits.
                sequence = torch.cat([sequence, sequence], dim=0)
                tr_seq = torch.cat([tr_seq, tr_seq], dim=0)
            all_output = model(sequence, tr_seq, conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_output, uncond_output = all_output.split(B, dim=0)  # [B, K, T, card]
                output = uncond_output + (cond_output - uncond_output) * cfg_coef
            else:
                output = all_output

        return output.transpose(1, 2)[..., -1].unsqueeze(-1)

    @torch.no_grad()
    def generate(
        self,
        prompt: tp.Optional[torch.Tensor] = None,
        conditions: tp.List[ConditioningAttributes] = [],
        tr: tp.List[float] = [],
        num_samples: tp.Optional[int] = None,
        max_gen_len: int = 256,
        use_sampling: bool = False,
        cfg_coef: tp.Optional[float] = None,
        two_step_cfg: tp.Optional[bool] = None,
        remove_prompts: bool = False,
        check: bool = False,
        callback: tp.Optional[tp.Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistent.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]

        cfg_conditions: CFGConditions
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            if two_step_cfg:
                cfg_conditions = (
                    self.condition_provider(self.condition_provider.tokenize(conditions)),
                    self.condition_provider(self.condition_provider.tokenize(null_conditions)),
                )
            else:
                conditions = conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions)
                cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        start_offset = T
        assert start_offset < max_gen_len

        # this token is used as default value for codes that are not generated yet
        unknown_token = -1

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        gen_bolds = torch.full((num_samples, self.space_dim, max_gen_len), unknown_token, device=device).to(
            dtype=prompt.dtype
        )
        gen_bolds[..., :start_offset] = prompt
        gen_bolds, _ = self.add_start_emb(gen_bolds)
        max_gen_len += 1  # add start embedding
        tr_tensor = self.tr_preprocess(tr, max_gen_len, [max_gen_len] * num_samples).to(device)
        start_offset_sequence = start_offset + 1  # add start embedding
        assert start_offset_sequence is not None

        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_bolds.shape[-1]
            for offset in range(start_offset_sequence, gen_sequence_len):
                # get current sequence (note that the streaming API is providing the caching over previous offsets)
                curr_sequence = gen_bolds[..., prev_offset:offset]
                curr_tr = tr_tensor[..., prev_offset:offset]
                # sample next token from the model, next token shape is [B, K, 1]
                next_seq = self._sample_next_token(
                    curr_sequence,
                    curr_tr,
                    cfg_conditions,
                    unconditional_state,
                    use_sampling=False,
                    cfg_coef=cfg_coef,
                )
                gen_bolds[..., offset : offset + 1] = next_seq
                prev_offset = offset
                if callback is not None:
                    callback(
                        1 + offset - start_offset_sequence,
                        gen_sequence_len - start_offset_sequence,
                    )
        unconditional_state.clear()

        out_start_offset = start_offset if remove_prompts else 1
        gen_bolds = gen_bolds[..., out_start_offset:max_gen_len]

        return gen_bolds
