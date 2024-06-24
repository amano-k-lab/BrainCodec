# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import typing as tp
from pathlib import Path

import flashy
import omegaconf
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from .. import models
from ..data.audio_dataset import AudioDataset
from ..data.fMRI_dataset import SegmentInfo
from ..models.builders import get_lm_model
from ..models.loaders import load_lm_model_ckpt
from ..solvers.compression_fMRI import CompressionfMRISolver
from ..utils import checkpoint
from ..utils.samples.manager import SampleManagerfMRI
from ..utils.utils import get_dataset_from_loader
from . import builders
from .musicgen import MusicGenSolver


class CSMrawfMRI(MusicGenSolver):
    """Solver for MusicGen training task.

    Used in: https://arxiv.org/abs/2306.05284
    """

    DATASET_TYPE: builders.DatasetType = builders.DatasetType.FMRI

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        # easier access to sampling parameters
        self.generation_params = {}
        self._best_metric_name: tp.Optional[str] = "l1"

        if cfg.compression_model_checkpoint is not None:
            self.logger.info("Loading compression model from checkpoint")
            self.codec_model = CompressionfMRISolver.model_from_checkpoint(
                cfg.compression_model_checkpoint, device=self.device
            )
            assert self.codec_model.tr_emb_pad_id == self.model.tr_emb_pad_id
        else:
            self.logger.info("No compression model checkpoint provided")
            self.codec_model = None

    def get_formatter(self, stage_name: str) -> flashy.Formatter:
        return flashy.Formatter(
            {
                "lr": ".2E",
                "l1": ".3f",
                "grad_norm": ".3E",
            },
        )

    def build_model(self) -> None:
        """Instantiate models and optimizer."""
        # instantiate LM model
        self.model: models.RawLMModel = models.builders.get_lm_model(self.cfg).to(self.device)
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
    def get_reconstructed_fMRI(self, bold: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
        output = self.codec_model.encode(bold, tr)
        if isinstance(output, torch.Tensor):
            rec = self.codec_model.decode(output, tr, noise_disable=True, use_layer=self.cfg.use_layer)
        elif len(output) == 2:
            codes, logvar = output
            rec = self.codec_model.decode(codes, tr, logvar, noise_disable=True, use_layer=self.cfg.use_layer)
        elif len(output) == 3:
            codes, mu, logvar = output
            rec = self.codec_model.decode(codes, tr, mu, logvar, noise_disable=True, use_layer=self.cfg.use_layer)
        else:
            raise RuntimeError("Only EncodecfMRIModel model is supported.")
        return rec

    def _compute_l1_loss(
        self,
        model_output: torch.Tensor,
        bold: torch.Tensor,
        bold_mask: torch.Tensor,
    ) -> torch.Tensor:
        model_output = model_output.contiguous().view(-1)
        bold = bold.contiguous().view(-1)
        bold_mask = bold_mask.contiguous().view(-1)
        return F.l1_loss(model_output[bold_mask], bold[bold_mask])

    def run_step(
        self,
        idx: int,
        batch: tp.Tuple[torch.Tensor, tp.List[SegmentInfo]],
        metrics: dict,
    ) -> dict:
        """Perform one training or valid step on a given batch."""
        check_synchronization_points = idx == 1 and self.device == "cuda"

        bold, bold_mask, bold_durations, tr = self._prepare_model_inputs(batch)
        if self.codec_model is not None:
            bold = self.get_reconstructed_fMRI(
                bold,
                self.model.tr_preprocess(tr, bold.shape[-1], bold_durations).to(bold.device),
            )
        self.deadlock_detect.update("tokens_and_conditions")

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode("warn")

        with self.autocast:
            model_output = self.model.compute_predictions(bold, bold_durations, tr, [])  # type: ignore
            l1 = self._compute_l1_loss(model_output.transpose(1, 2)[..., :-1], bold, bold_mask)
            loss = l1
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

        metrics["l1"] = l1

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
        bench_start = time.time()
        bold, bold_mask, bold_durations, tr = self._prepare_model_inputs(batch)
        if self.codec_model is not None:
            bold = self.get_reconstructed_fMRI(
                bold,
                self.model.tr_preprocess(tr, bold.shape[-1], bold_durations).to(bold.device),
            )

        # prepare audio prompt
        if prompt_duration is None:
            raise ValueError("Prompt duration must be specified for fMRI")
        else:
            assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
            prompt_bold = bold[..., :prompt_duration]
            prompt_mask = torch.ones_like(bold, dtype=torch.bool, device=prompt_bold.device)
            prompt_mask[:, :, :prompt_duration] = False

        # get audio tokens from compression model
        num_samples = None
        prompt_bold = prompt_bold.to(self.device)

        # generate by sampling from the LM
        with self.autocast:
            gen_bold = self.model.generate(
                prompt_bold,
                [],
                tr,
                max_gen_len=gen_duration,
                num_samples=num_samples,
                **self.generation_params,
            )

        # generate audio from tokens
        bench_end = time.time()

        # Calc loss
        gen_loss = self._compute_l1_loss(gen_bold, bold, bold_mask & prompt_mask)

        gen_outputs = {
            "rtf": (bench_end - bench_start) / gen_duration,
            "ref_bold": bold,
            "gen_bold": gen_bold,
            "prompt_bold": prompt_bold,
            "l1": gen_loss,
            "tr": tr,
        }
        return gen_outputs

    def generate_fMRI(self) -> dict:
        """Audio generation stage."""
        generate_stage_name = f"{self.current_stage}"
        sample_manager = SampleManagerfMRI(self.xp, map_reference_to_sample_id=True)
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

        metrics: dict = {}
        average = flashy.averager()
        for batch in lp:
            # metadata for sample manager
            if self.cfg.generate.lm.unprompted_samples:
                raise ValueError("Unprompted samples are not supported for fMRI")

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
                    None,
                    prompt_bolds=None,
                    ground_truth_bolds=gen_outputs["ref_bold"].cpu(),
                )
                metrics["rtf"] = gen_outputs["rtf"]
                metrics["l1"] = gen_outputs["l1"]
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

    @staticmethod
    def model_from_checkpoint(
        checkpoint_path: tp.Union[Path, str],
        device: tp.Union[torch.device, str] = "cpu",
    ) -> models.LMModel:
        _checkpoint_path = checkpoint.resolve_checkpoint_path(str(checkpoint_path), use_fsdp=False)
        pkg = load_lm_model_ckpt(_checkpoint_path)
        cfg = OmegaConf.create(pkg["xp.cfg"])
        cfg.device = str(device)
        if cfg.device == "cpu":
            cfg.dtype = "float32"
        else:
            cfg.dtype = "float16"
        model = get_lm_model(cfg)
        if pkg["fsdp_best_state"]:
            model.load_state_dict(pkg["fsdp_best_state"]["model"])
        else:
            model.load_state_dict(pkg["best_state"]["model"])
        model.eval()
        model.cfg = cfg
        return model
