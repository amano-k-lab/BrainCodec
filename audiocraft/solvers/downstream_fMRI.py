# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
import typing as tp
from pathlib import Path

import flashy
import omegaconf
import torch
from torch import nn

from .. import models
from ..data.fMRI_dataset import SegmentInfo
from ..solvers.compression_fMRI import CompressionfMRISolver
from ..utils import checkpoint
from ..utils.utils import get_pool_executor
from . import base, builders

logger = logging.getLogger(__name__)


class DownstreamfMRISolver(base.StandardSolver):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.rng: torch.Generator  # set at each epoch
        self.aux_losses = nn.ModuleDict()
        self.info_losses = nn.ModuleDict()
        assert not cfg.fsdp.use, "FSDP not supported by CompressionSolver."
        loss_weights = dict()
        for loss_name, weight in self.cfg.losses.items():
            if weight > 0:
                self.aux_losses[loss_name] = builders.get_loss(loss_name, self.cfg)
                loss_weights[loss_name] = weight
            else:
                self.info_losses[loss_name] = builders.get_loss(loss_name, self.cfg)
        self.balancer = builders.get_balancer(loss_weights, self.cfg.balancer)

        if cfg.compression_model_checkpoint is not None:
            logger.info("Loading compression model from checkpoint")
            self.codec_model = CompressionfMRISolver.model_from_checkpoint(
                cfg.compression_model_checkpoint, device=self.device
            )
        else:
            logger.info("No compression model checkpoint provided")
            self.codec_model = None

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        # best model is the last for the compression model
        return None

    def build_model(self):
        """Instantiate model and optimizer."""
        # Model and optimizer
        self.model = models.builders.get_downstream_fMRI_model(self.cfg).to(self.device)
        self.optimizer = builders.get_optimizer(self.model.parameters(), self.cfg.optim)
        self.lr_scheduler = builders.get_lr_scheduler(self.optimizer, self.cfg.schedule, self.total_updates)
        self.register_stateful("model", "optimizer", "lr_scheduler")
        self.register_best_state("model")
        self.register_ema("model")

    def build_dataloaders(self):
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_fMRI_datasets(self.cfg)

    def show(self):
        """Show the compression model and employed adversarial loss."""
        self.log_model_summary(self.model)
        self.logger.info("Auxiliary losses:")
        self.logger.info(self.aux_losses)
        self.logger.info("Info losses:")
        self.logger.info(self.info_losses)

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

    @torch.no_grad()
    def _preprocess_tr(
        self, batch: tp.Tuple[torch.Tensor, SegmentInfo], return_raw_tr=False
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess training batch."""
        x, infos = batch
        x = x.to(self.device)
        tr = torch.zeros(x.size(0), x.size(-1), device=self.device).long()
        raw_trs = []
        labels = []
        for i, info in enumerate(infos):
            _tr = info.tr
            labels.append(info.meta.label)
            raw_trs.append(_tr)
            fMRI_duration = info.duration
            _tr_tensor = torch.arange(0, fMRI_duration, device=self.device) * _tr

            # fMRI_duration が x.size(-1) より大きい場合は、テンソルをクロップ
            if fMRI_duration > x.size(-1):
                _tr_tensor = _tr_tensor[: x.size(-1)]
            else:
                # x.size(-1) 未満の場合は、残りの部分をゼロで埋める
                pad_length = x.size(-1) - fMRI_duration
                pad_id = info.max_tr + info.tr_precision
                _tr_tensor = torch.cat((_tr_tensor, pad_id * torch.ones(pad_length, device=self.device)))

            # ID にする
            _tr_tensor = (_tr_tensor / info.tr_precision).floor()

            # 結果を tr テンソルに追加
            tr[i] = _tr_tensor

        labels = torch.tensor(labels, device=self.device).long()

        if return_raw_tr is True:
            return x, tr, raw_trs, labels
        return x, tr, labels

    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        x, tr, labels = self._preprocess_tr(batch)
        if self.codec_model is not None:
            x = self.get_reconstructed_fMRI(x, tr)
        y = labels
        y_pred = self.model(x)

        balanced_losses: dict = {}
        # auxiliary losses
        for loss_name, criterion in self.aux_losses.items():
            loss = criterion(y_pred, y)
            balanced_losses[loss_name] = loss

        # weighted losses
        metrics.update(balanced_losses)

        if self.is_training:
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            # balancer losses backward, returns effective training loss
            # with effective weights at the current batch.
            metrics["g_loss"] = self.balancer.backward(balanced_losses, y_pred)
            # add metrics corresponding to weight ratios
            metrics.update(self.balancer.metrics)
            ratio2 = sum(p.grad.data.norm(p=2).pow(2) for p in self.model.parameters() if p.grad is not None)
            assert isinstance(ratio2, torch.Tensor)
            metrics["ratio2"] = ratio2.sqrt()

            # optim
            flashy.distrib.sync_model(self.model)
            if self.cfg.optim.max_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.max_norm)
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # informative losses only
        info_losses: dict = {}
        with torch.no_grad():
            for loss_name, criterion in self.info_losses.items():
                loss = criterion(y_pred, y)
                info_losses[loss_name] = loss

        metrics.update(info_losses)

        return metrics

    def run_epoch(self):
        # reset random seed at the beginning of the epoch
        self.rng = torch.Generator()
        self.rng.manual_seed(1234 + self.epoch)
        # run epoch
        super().run_epoch()

    def evaluate(self):
        self.model.eval()
        evaluate_stage_name = str(self.current_stage)

        loader = self.dataloaders["evaluate"]
        updates = len(loader)
        lp = self.log_progress(
            f"{evaluate_stage_name} inference",
            loader,
            total=updates,
            updates=self.log_updates,
        )
        average = flashy.averager()

        pendings = []
        ctx = multiprocessing.get_context("spawn")
        with get_pool_executor(self.cfg.evaluate.num_workers, mp_context=ctx) as pool:
            for idx, batch in enumerate(lp):
                x, tr, labels = self._preprocess_tr(batch)
                if self.codec_model is not None:
                    x = self.get_reconstructed_fMRI(x, tr)
                with torch.no_grad():
                    y_pred = self.model(x)

                y = labels.cpu()  # should already be on CPU but just in case
                y_pred = y_pred.cpu()  # should already be on CPU but just in case
                pendings.append(pool.submit(evaluate_fMRI_acc, y_pred, y, self.cfg))  # Need to change this

            metrics_lp = self.log_progress(f"{evaluate_stage_name} metrics", pendings, updates=self.log_updates)
            for pending in metrics_lp:
                metrics = pending.result()
                metrics = average(metrics)

        metrics = flashy.distrib.average_metrics(metrics, len(loader))
        return metrics

    def generate(self):
        self.logger.info("Generating skipped for DownstreamfMRISolver.")
        flashy.distrib.barrier()

    def load_from_pretrained(self, name: str) -> dict:
        # 使われてないのでOK
        raise NotImplementedError("Pretrained models not supported by DownstreamfMRISolver.")

    @staticmethod
    def model_from_checkpoint(
        checkpoint_path: tp.Union[Path, str],
        device: tp.Union[torch.device, str] = "cpu",
        need_cfg: bool = False,
    ) -> models.CompressionModel:
        checkpoint_path = str(checkpoint_path)
        logger = logging.getLogger(__name__)
        logger.info(f"Loading compression model from checkpoint: {checkpoint_path}")
        _checkpoint_path = checkpoint.resolve_checkpoint_path(checkpoint_path, use_fsdp=False)
        assert _checkpoint_path is not None, f"Could not resolve compression model checkpoint path: {checkpoint_path}"
        state = checkpoint.load_checkpoint(_checkpoint_path)
        assert state is not None and "xp.cfg" in state, f"Could not load compression model from ckpt: {checkpoint_path}"
        cfg = state["xp.cfg"]
        cfg.device = device
        downstream_fMRI_model = models.builders.get_downstream_fMRI_model(cfg).to(device)
        assert (
            downstream_fMRI_model.space_dim == cfg.space_dim
        ), "Compression model sample rate should match"  # Need to change this

        assert "best_state" in state and state["best_state"] != {}
        assert "exported" not in state, "When loading an exported checkpoint, use the //pretrained/ prefix."
        downstream_fMRI_model.load_state_dict(state["best_state"]["model"])
        downstream_fMRI_model.eval()
        logger.info("downstream_fMRI model loaded!")
        if need_cfg is True:
            return downstream_fMRI_model, cfg
        return downstream_fMRI_model

    @staticmethod
    def wrapped_model_from_checkpoint(
        cfg: omegaconf.DictConfig,
        checkpoint_path: tp.Union[Path, str],
        device: tp.Union[torch.device, str] = "cpu",
    ) -> models.CompressionModel:
        # 使われていないっぽいので OK
        raise NotImplementedError("Wrapped model not supported by CompressionSolver.")


def evaluate_fMRI_acc(y_pred: torch.Tensor, y: torch.Tensor, cfg: omegaconf.DictConfig) -> dict:
    metrics = {}
    # y_predから最も可能性の高いクラスを選択
    if cfg.evaluate.metrics.acc:
        accuracy_score = builders.get_loss("accuracy", cfg)
        metrics["accuracy"] = accuracy_score(y_pred, y)
    if cfg.evaluate.metrics.f1:
        f1_score = builders.get_loss("f1", cfg)
        f1 = f1_score(y_pred, y)
        metrics["f1"] = f1

    return metrics
