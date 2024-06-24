# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import typing as tp

import flashy
import omegaconf
import torch

from .. import models
from ..data.audio_dataset import AudioDataset
from ..data.fMRI_dataset import SegmentInfo
from ..modules.conditioners import JointEmbedCondition, WavCondition
from ..solvers.compression_fMRI import CompressionfMRISolver
from ..utils.samples.manager import SampleManagerfMRI
from ..utils.utils import get_dataset_from_loader, is_jsonable
from . import builders
from .musicgen import MusicGenSolver


class CSMfMRI(MusicGenSolver):
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.FMRI

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        # easier access to sampling parameters
        self.generation_params = {}
        self._best_metric_name: tp.Optional[str] = "ce"

        if cfg.compression_model_checkpoint is not None:
            self.logger.info("Loading compression model from checkpoint")
            self.compression_model = CompressionfMRISolver.model_from_checkpoint(
                cfg.compression_model_checkpoint, device=self.device
            )
            assert self.compression_model.tr_emb_pad_id == self.model.tr_emb_pad_id
        else:
            self.logger.info("No compression model checkpoint provided")
            self.compression_model = None

    def get_formatter(self, stage_name: str) -> flashy.Formatter:
        return flashy.Formatter(
            {
                "ce": ".3f",
                "ppl": ".3f",
                "grad_norm": ".3E",
            },
            exclude_keys=["ce_q*", "ppl_q*"],
        )

    def build_model(self) -> None:
        """Instantiate models and optimizer."""
        # instantiate LM model
        self.model: models.LMModelfMRI = models.builders.get_lm_model(self.cfg).to(self.device)
        if self.cfg.fsdp.use:
            assert not self.cfg.autocast, "Cannot use autocast with fsdp"
            self.model = self.wrap_with_fsdp(self.model)
        self.register_ema("model")
        # initialize optimization
        self.optimizer = builders.get_optimizer(builders.get_optim_parameter_groups(self.model), self.cfg.optim)
        self.lr_scheduler = builders.get_lr_scheduler(self.optimizer, self.cfg.schedule, self.total_updates)
        self.register_stateful("model", "optimizer", "lr_scheduler")
        self.register_best_state("model")
        self.autocast_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[self.cfg.autocast_dtype]
        self.scaler: tp.Optional[torch.cuda.amp.GradScaler] = None
        if self.cfg.fsdp.use:
            need_scaler = self.cfg.fsdp.param_dtype == "float16"
        else:
            need_scaler = self.cfg.autocast and self.autocast_dtype is torch.float16
        if need_scaler:
            if self.cfg.fsdp.use:
                from torch.distributed.fsdp.sharded_grad_scaler import \
                    ShardedGradScaler

                self.scaler = ShardedGradScaler()  # type: ignore
            else:
                self.scaler = torch.cuda.amp.GradScaler()
            self.register_stateful("scaler")

    def build_dataloaders(self) -> None:
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_fMRI_datasets(self.cfg)

    def show(self) -> None:
        """Show the compression model and LM model."""
        self.logger.info("LM model:")
        self.log_model_summary(self.model)

    @torch.no_grad()
    def _prepare_model_inputs(
        self, batch: tp.Tuple[torch.Tensor, SegmentInfo]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.List[int], tp.List[float]]:
        bold, infos = batch
        bold = bold.to(self.device)  # [B, dim, T]

        bold_durations = [info.duration for info in infos]
        tr = [info.tr for info in infos]

        mask = torch.ones_like(bold, dtype=torch.bool, device=bold.device)
        for b_d in bold_durations:
            mask[:, :, b_d:] = False

        return bold, mask, bold_durations, tr

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
        self,
        batch: tp.Tuple[torch.Tensor, tp.List[SegmentInfo]],
        tr: torch.Tensor,
        check_synchronization_points: bool = False,
    ) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
        bold, infos = batch
        bold = bold.to(self.device)
        bold_tokens = None

        # prepare attributes
        attributes = [info.to_condition_attributes() for info in infos]
        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)

        # Now we should be synchronization free.
        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("warn")

        if bold_tokens is None:
            with torch.no_grad():
                bold_tokens, *_ = self.compression_model.encode(bold, tr)

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(bold_tokens, dtype=torch.bool, device=bold_tokens.device)
        # replace encodec tokens from padded audio with special_token_id
        if self.cfg.tokens.padding_with_special_token:
            bold_tokens = bold_tokens.clone()
            padding_mask = padding_mask.clone()
            B, K, T_s = bold_tokens.shape
            for i in range(B):
                valid_tokens = infos[i].duration
                # take the last token generated from actual audio frames (non-padded audio)
                bold_tokens[i, :, valid_tokens:] = self.model.special_token_id
                padding_mask[i, :, valid_tokens:] = 0

        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("default")

        return condition_tensors, bold_tokens, padding_mask

    def run_step(
        self,
        idx: int,
        batch: tp.Tuple[torch.Tensor, tp.List[SegmentInfo]],
        metrics: dict,
    ) -> dict:
        """Perform one training or valid step on a given batch."""
        check_synchronization_points = idx == 1 and self.device == "cuda"

        bold, _, bold_durations, tr = self._prepare_model_inputs(batch)
        if self.compression_model is not None:
            (
                condition_tensors,
                bold_tokens,
                padding_mask,
            ) = self._prepare_tokens_and_attributes(
                batch,
                self.model.tr_preprocess(tr, bold.shape[-1], bold_durations).to(bold.device),
                check_synchronization_points,
            )
        self.deadlock_detect.update("tokens_and_conditions")

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode("warn")

        with self.autocast:
            model_output = self.model.compute_predictions_for_fmri(
                bold_tokens, [d * 3 for d in bold_durations], tr, condition_tensors
            )  # type: ignore
            logits = model_output.logits
            mask = padding_mask & model_output.mask
            ce, ce_per_codebook = self._compute_cross_entropy(logits, bold_tokens, mask)
            loss = ce
        self.deadlock_detect.update("loss")

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode("default")

        if self.is_training:
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            self.deadlock_detect.update("scale")
            if self.cfg.fsdp.use:
                loss.backward()
                flashy.distrib.average_tensors(self.model.buffers())
            elif self.cfg.optim.eager_sync:
                with flashy.distrib.eager_sync_model(self.model):
                    loss.backward()
            else:
                # this should always be slower but can be useful
                # for weird use cases like multiple backwards.
                loss.backward()
                flashy.distrib.sync_model(self.model)
            self.deadlock_detect.update("backward")

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.cfg.optim.max_norm:
                if self.cfg.fsdp.use:
                    metrics["grad_norm"] = self.model.clip_grad_norm_(self.cfg.optim.max_norm)  # type: ignore
                else:
                    metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.max_norm
                    )
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.deadlock_detect.update("optim")
            if self.scaler is not None:
                scale = self.scaler.get_scale()
                metrics["grad_scale"] = scale
            if not loss.isfinite().all():
                raise RuntimeError("Model probably diverged.")

        metrics["ce"] = ce
        metrics["ppl"] = torch.exp(ce)
        for k, ce_q in enumerate(ce_per_codebook):
            metrics[f"ce_q{k + 1}"] = ce_q
            metrics[f"ppl_q{k + 1}"] = torch.exp(ce_q)

        return metrics

    @torch.no_grad()
    def run_generate_step(
        self,
        batch: tp.Tuple[torch.Tensor, tp.List[SegmentInfo]],
        gen_duration: float,
        prompt_duration: tp.Optional[float] = None,
        remove_prompt: bool = False,
        **generation_params,
    ) -> dict:
        """Run generate step on a batch of optional audio tensor and corresponding attributes.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]):
            use_prompt (bool): Whether to do audio continuation generation with prompt from audio batch.
            gen_duration (float): Target audio duration for the generation.
            prompt_duration (float, optional): Duration for the audio prompt to use for continuation.
            remove_prompt (bool, optional): Whether to remove the prompt from the generated audio.
            generation_params: Additional generation parameters.
        Returns:
            gen_outputs (dict): Generation outputs, consisting in audio, audio tokens from both the generation
                and the prompt along with additional information.
        """
        bench_start = time.time()
        _, _, bold_durations, tr = self._prepare_model_inputs(batch)
        bold, meta = batch
        # prepare attributes
        attributes = [x.to_condition_attributes() for x in meta]
        # TODO: Add dropout for chroma?

        # prepare audio prompt
        if prompt_duration is None:
            prompt_bold = None
        else:
            assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
            prompt_bold = bold[..., :prompt_duration]
            prompt_bold_durations = [min(d, prompt_duration) for d in bold_durations]

        # get audio tokens from compression model
        if prompt_bold is None or prompt_bold.nelement() == 0:
            num_samples = len(attributes)
            prompt_tokens = None
        else:
            num_samples = None
            prompt_bold = prompt_bold.to(self.device)
            prompt_tokens, *_ = self.compression_model.encode(
                prompt_bold,
                self.model.tr_preprocess(tr, prompt_duration, prompt_bold_durations).to(prompt_bold.device),
            )

        # generate by sampling from the LM
        with self.autocast:
            # bold token 自体は encode により 3倍になることに注意
            total_gen_len = gen_duration * 3
            gen_tokens = self.model.generate(
                prompt_tokens,
                attributes,
                tr,
                max_gen_len=total_gen_len,
                num_samples=num_samples,
                **self.generation_params,
            )

        # generate audio from tokens
        assert gen_tokens.dim() == 3
        gen_bold = self.compression_model.decode(
            gen_tokens,
            self.model.tr_preprocess(tr, gen_duration, [gen_duration] * len(bold_durations)).to(prompt_bold.device),
            None,
            None,
            noise_disable=True,
        )

        bench_end = time.time()
        gen_outputs = {
            "rtf": (bench_end - bench_start) / gen_duration,
            "ref_bold": bold,
            "gen_bold": gen_bold,
            "gen_tokens": gen_tokens,
            "prompt_bold": prompt_bold,
            "prompt_tokens": prompt_tokens,
            "tr": tr,
        }
        return gen_outputs

    def generate_fMRI(self) -> dict:
        """Audio generation stage."""
        generate_stage_name = f"{self.current_stage}"
        sample_manager = SampleManagerfMRI(self.xp)
        self.logger.info(f"Generating samples in {sample_manager.base_folder}")
        loader = self.dataloaders["generate"]
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        dataset = get_dataset_from_loader(loader)
        dataset_duration = dataset.segment_duration
        assert dataset_duration is not None
        assert isinstance(dataset, AudioDataset)
        target_duration = self.cfg.generate.lm.gen_duration
        prompt_duration = self.cfg.generate.lm.prompt_duration
        if target_duration is None:
            target_duration = dataset_duration
        if prompt_duration is None:
            prompt_duration = dataset_duration // 4
        assert prompt_duration < dataset_duration, (
            f"Specified prompt duration ({prompt_duration}s) is longer",
            f" than reference audio duration ({dataset_duration}s)",
        )

        def get_hydrated_conditions(meta: tp.List[SegmentInfo]):
            hydrated_conditions = []
            for sample in [x.to_condition_attributes() for x in meta]:
                cond_dict = {}
                for cond_type in sample.__annotations__.keys():
                    for cond_key, cond_val in getattr(sample, cond_type).items():
                        if cond_key not in self.model.condition_provider.conditioners.keys():
                            continue
                        if is_jsonable(cond_val):
                            cond_dict[cond_key] = cond_val
                        elif isinstance(cond_val, WavCondition):
                            cond_dict[cond_key] = cond_val.path
                        elif isinstance(cond_val, JointEmbedCondition):
                            cond_dict[cond_key] = cond_val.text  # only support text at inference for now
                        else:
                            # if we reached this point, it is not clear how to log the condition
                            # so we just log the type.
                            cond_dict[cond_key] = str(type(cond_val))
                            continue
                hydrated_conditions.append(cond_dict)
            return hydrated_conditions

        metrics: dict = {}
        average = flashy.averager()
        for batch in lp:
            bold, meta = batch
            # metadata for sample manager
            hydrated_conditions = get_hydrated_conditions(meta)
            sample_generation_params = {
                **{f"classifier_free_guidance_{k}": v for k, v in self.cfg.classifier_free_guidance.items()},
                **self.generation_params,
            }
            if self.cfg.generate.lm.unprompted_samples:
                raise NotImplementedError("Unprompted samples are not supported for fMRI generation")

            if self.cfg.generate.lm.prompted_samples:
                gen_outputs = self.run_generate_step(
                    batch,
                    gen_duration=target_duration,
                    prompt_duration=prompt_duration,
                    **self.generation_params,
                )
                sample_manager.add_samples(
                    gen_outputs["gen_bold"].cpu(),
                    gen_outputs["tr"],
                    self.epoch,
                    hydrated_conditions,
                    prompt_bolds=None,  # 未対応
                    # segment size が 64 に満たないものも当然あることに注意
                    ground_truth_bolds=gen_outputs["ref_bold"].cpu(),
                )
            metrics["rtf"] = gen_outputs["rtf"]
            metrics = average(metrics)

        flashy.distrib.barrier()
        return metrics

    def generate(self) -> dict:
        """Generate stage."""
        self.model.eval()
        with torch.no_grad():
            return self.generate_fMRI()

    def evaluate(self) -> dict:
        """Evaluate stage."""
        self.model.eval()
        with torch.no_grad():
            metrics: dict = {}
            if self.cfg.evaluate.metrics.base:
                metrics.update(self.common_train_valid("evaluate"))
            return metrics
