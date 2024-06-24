# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import flashy
import omegaconf
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from .. import models
from ..data.fMRI_dataset import SegmentInfo
from ..models.builders import get_lm_model
from ..models.loaders import load_lm_model_ckpt
from ..solvers.compression_fMRI import CompressionfMRISolver
from ..utils import checkpoint
from . import builders
from .CSM_fMRI import CSMfMRI


class DownstreamCSMfMRI(CSMfMRI):
    """Solver for MusicGen training task.

    Used in: https://arxiv.org/abs/2306.05284
    """

    DATASET_TYPE: builders.DatasetType = builders.DatasetType.FMRI

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        # easier access to sampling parameters
        self._best_metric_name: tp.Optional[str] = "cross_entropy"

        self.info_losses = {
            "accuracy": builders.get_loss("accuracy", self.cfg),
            "f1": builders.get_loss("f1", self.cfg),
        }

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
                "cross_entropy": ".3f",
                "accuracy": ".3f",
                "f1": ".3f",
                "grad_norm": ".3E",
            },
        )

    def build_model(self) -> None:
        """Instantiate models and optimizer."""
        # instantiate LM model
        self.model: models.DownstreamLMModel = models.builders.get_lm_model(self.cfg).to(self.device)
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

        # load pretrained weight
        if self.cfg.CSM_pretrained_checkpoint is not None:
            _checkpoint_path = checkpoint.resolve_checkpoint_path(
                str(self.cfg.CSM_pretrained_checkpoint), use_fsdp=False
            )
            state = checkpoint.load_checkpoint(_checkpoint_path)
            if self.cfg.fsdp.use:
                raise NotImplementedError("FSDP not supported for CSM pretrained checkpoint")
            else:
                model_state_dict = self.model.state_dict()
                loaded_state_dict = state["best_state"]["model"]

                # stateにはあるがmodelにはないキーがあるかチェック
                missing_keys = set(loaded_state_dict.keys()) - set(model_state_dict.keys())
                if missing_keys:
                    raise KeyError(f"Keys {missing_keys} found in state but not in model")

                # modelの重みを更新
                model_state_dict.update(loaded_state_dict)
                self.model.load_state_dict(model_state_dict)

                # ロガーに報告
                self.logger.info("Model weights loaded successfully")
        else:
            self.logger.info("No pretrained checkpoint provided")

    def show(self) -> None:
        """Show the compression model and LM model."""
        if self.codec_model is not None:
            self.logger.info("Compression model:")
            self.log_model_summary(self.codec_model)
        self.logger.info("LM model:")
        self.log_model_summary(self.model)

    def _compute_ce_loss(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(model_output, labels)

    @torch.no_grad()
    def _prepare_model_inputs(
        self, batch: tp.Tuple[torch.Tensor, SegmentInfo]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.List[int], tp.List[float], torch.Tensor]:
        bold, infos = batch
        bold = bold.to(self.device)  # [B, dim, T]

        bold_durations = [info.duration for info in infos]
        tr = [info.tr for info in infos]
        labels = torch.tensor([info.meta.label for info in infos]).long().to(bold.device)

        mask = torch.ones_like(bold, dtype=torch.bool, device=bold.device)
        for b_d in bold_durations:
            mask[:, :, b_d:] = False

        return bold, mask, bold_durations, tr, labels

    def run_step(
        self,
        idx: int,
        batch: tp.Tuple[torch.Tensor, tp.List[SegmentInfo]],
        metrics: dict,
    ) -> dict:
        """Perform one training or valid step on a given batch."""
        check_synchronization_points = idx == 1 and self.device == "cuda"

        bold, _, bold_durations, tr, labels = self._prepare_model_inputs(batch)
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
            )
            ce = self._compute_ce_loss(model_output, labels)
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

        metrics["cross_entropy"] = ce

        # informative losses only
        info_losses: dict = {}
        with torch.no_grad():
            for loss_name, criterion in self.info_losses.items():
                loss = criterion(model_output, labels)
                info_losses[loss_name] = loss

        metrics.update(info_losses)

        return metrics

    def generate(self) -> dict:
        """Generate stage."""
        raise NotImplementedError("Generate not implemented for DownstreamCSMrawfMRI")

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
