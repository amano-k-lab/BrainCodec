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

from .. import models, quantization
from ..data.fMRI_dataset import SegmentInfo
from ..utils import checkpoint
from ..utils.samples.manager import SampleManagerfMRI
from ..utils.utils import get_pool_executor
from . import builders
from .compression import CompressionSolver

logger = logging.getLogger(__name__)


class CompressionfMRISolver(CompressionSolver):
    def build_dataloaders(self):
        """Instantiate fMRI dataloaders for each stage."""
        self.dataloaders = builders.get_fMRI_datasets(self.cfg)

    @torch.no_grad()
    def _preprocess_tr(
        self, batch: tp.Tuple[torch.Tensor, SegmentInfo], return_raw_tr=False
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess training batch."""
        x, infos = batch
        x = x.to(self.device)
        tr = torch.zeros(x.size(0), x.size(-1), device=self.device).long()
        raw_trs = []
        for i, info in enumerate(infos):
            _tr = info.tr
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

        if return_raw_tr is True:
            return x, tr, raw_trs
        return x, tr

    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        """Perform one training or valid step on a given batch."""
        x, tr = self._preprocess_tr(batch)
        y = x.clone()

        qres = self.model(x, tr)
        assert isinstance(qres, quantization.QuantizedResult)
        y_pred = qres.x
        # Log bandwidth in kb/s
        metrics["bandwidth"] = qres.bandwidth.mean()

        if self.is_training:
            d_losses: dict = {}
            if len(self.adv_losses) > 0 and torch.rand(1, generator=self.rng).item() <= 1 / self.cfg.adversarial.every:
                for adv_name, adversary in self.adv_losses.items():
                    disc_loss = adversary.train_adv(y_pred, y)
                    d_losses[f"d_{adv_name}"] = disc_loss
                metrics["d_loss"] = torch.sum(torch.stack(list(d_losses.values())))
            metrics.update(d_losses)

        balanced_losses: dict = {}
        other_losses: dict = {}

        # penalty from quantization
        if qres.penalty is not None and qres.penalty.requires_grad:
            other_losses["penalty"] = qres.penalty  # penalty term from the quantizer

        # adversarial losses
        for adv_name, adversary in self.adv_losses.items():
            adv_loss, feat_loss = adversary(y_pred, y)
            balanced_losses[f"adv_{adv_name}"] = adv_loss
            balanced_losses[f"feat_{adv_name}"] = feat_loss

        # auxiliary losses
        for loss_name, criterion in self.aux_losses.items():
            loss = criterion(y_pred, y)
            balanced_losses[loss_name] = loss

        # weighted losses
        metrics.update(balanced_losses)
        metrics.update(other_losses)
        metrics.update(qres.metrics)

        if self.is_training:
            # backprop losses that are not handled by balancer
            other_loss = torch.tensor(0.0, device=self.device)
            if "penalty" in other_losses:
                other_loss += other_losses["penalty"]
            if other_loss.requires_grad:
                other_loss.backward(retain_graph=True)
                ratio1 = sum(p.grad.data.norm(p=2).pow(2) for p in self.model.parameters() if p.grad is not None)
                assert isinstance(ratio1, torch.Tensor)
                metrics["ratio1"] = ratio1.sqrt()

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
            self.optimizer.zero_grad()

        # informative losses only
        info_losses: dict = {}
        with torch.no_grad():
            for loss_name, criterion in self.info_losses.items():
                loss = criterion(y_pred, y)
                info_losses[loss_name] = loss

        metrics.update(info_losses)

        # aggregated GAN losses: this is useful to report adv and feat across different adversarial loss setups
        adv_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith("adv")]
        if len(adv_losses) > 0:
            metrics["adv"] = torch.sum(torch.stack(adv_losses))
        feat_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith("feat")]
        if len(feat_losses) > 0:
            metrics["feat"] = torch.sum(torch.stack(feat_losses))

        return metrics

    def evaluate(self):
        """Evaluate stage. Runs audio reconstruction evaluation."""
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
                x, tr = self._preprocess_tr(batch)
                with torch.no_grad():
                    qres = self.model(x, tr)

                y_pred = qres.x.cpu()
                y = batch[0].cpu()  # should already be on CPU but just in case
                pendings.append(pool.submit(evaluate_fMRI_reconstruction, y_pred, y, self.cfg))  # Need to change this

            metrics_lp = self.log_progress(f"{evaluate_stage_name} metrics", pendings, updates=self.log_updates)
            for pending in metrics_lp:
                metrics = pending.result()
                metrics = average(metrics)

        metrics = flashy.distrib.average_metrics(metrics, len(loader))
        return metrics

    def generate(self):
        """Generate stage."""
        self.model.eval()
        stats = None
        normalize_mean = self.dataloaders["train"].dataset.normalize_mean
        normalize_std = self.dataloaders["train"].dataset.normalize_std
        if normalize_mean is not None:
            stats = {
                "mean": normalize_mean,
                "std": normalize_std,
            }
        sample_manager = SampleManagerfMRI(self.xp, map_reference_to_sample_id=True)
        generate_stage_name = str(self.current_stage)

        loader = self.dataloaders["generate"]
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        for batch in lp:
            reference, tr, tr_raw = self._preprocess_tr(batch, return_raw_tr=True)
            with torch.no_grad():
                qres = self.model(reference, tr)
            assert isinstance(qres, quantization.QuantizedResult)

            reference = reference.cpu()
            estimate = qres.x.cpu()
            sample_manager.add_samples(estimate, tr_raw, self.epoch, ground_truth_bolds=reference, stats=stats)

        flashy.distrib.barrier()

    @staticmethod
    def model_from_checkpoint(
        checkpoint_path: tp.Union[Path, str],
        device: tp.Union[torch.device, str] = "cpu",
    ) -> models.CompressionModel:
        """Instantiate a CompressionModel from a given checkpoint path or dora sig.
        This method is a convenient endpoint to load a CompressionModel to use in other solvers.

        Args:
            checkpoint_path (Path or str): Path to checkpoint or dora sig from where the checkpoint is resolved.
                This also supports pre-trained models by using a path of the form //pretrained/NAME.
                See `model_from_pretrained` for a list of supported pretrained models.
            use_ema (bool): Use EMA variant of the model instead of the actual model.
            device (torch.device or str): Device on which the model is loaded.
        """
        checkpoint_path = str(checkpoint_path)
        if checkpoint_path.startswith("//pretrained/"):
            name = checkpoint_path.split("/", 3)[-1]
            return models.CompressionModel.get_pretrained(name, device)
        logger = logging.getLogger(__name__)
        logger.info(f"Loading compression model from checkpoint: {checkpoint_path}")
        _checkpoint_path = checkpoint.resolve_checkpoint_path(checkpoint_path, use_fsdp=False)
        assert _checkpoint_path is not None, f"Could not resolve compression model checkpoint path: {checkpoint_path}"
        state = checkpoint.load_checkpoint(_checkpoint_path)
        assert state is not None and "xp.cfg" in state, f"Could not load compression model from ckpt: {checkpoint_path}"
        cfg = state["xp.cfg"]
        cfg.device = device
        compression_model = models.builders.get_compression_model(cfg).to(device)
        assert (
            compression_model.space_dim == cfg.space_dim
        ), "Compression model sample rate should match"  # Need to change this

        assert "best_state" in state and state["best_state"] != {}
        assert "exported" not in state, "When loading an exported checkpoint, use the //pretrained/ prefix."
        compression_model.load_state_dict(state["best_state"]["model"])
        compression_model.eval()
        logger.info("Compression model loaded!")
        return compression_model

    @staticmethod
    def wrapped_model_from_checkpoint(
        cfg: omegaconf.DictConfig,
        checkpoint_path: tp.Union[Path, str],
        device: tp.Union[torch.device, str] = "cpu",
    ) -> models.CompressionModel:
        """Instantiate a wrapped CompressionfMRIModel from a given checkpoint path or dora sig.

        Args:
            cfg (omegaconf.DictConfig): Configuration to read from for wrapped mode.
            checkpoint_path (Path or str): Path to checkpoint or dora sig from where the checkpoint is resolved.
            use_ema (bool): Use EMA variant of the model instead of the actual model.
            device (torch.device or str): Device on which the model is loaded.
        """
        compression_model = CompressionfMRISolver.model_from_checkpoint(checkpoint_path, device)  # Need to change this
        compression_model = models.builders.get_wrapped_compression_model(compression_model, cfg)
        return compression_model


def evaluate_fMRI_reconstruction(y_pred: torch.Tensor, y: torch.Tensor, cfg: omegaconf.DictConfig) -> dict:
    """Audio reconstruction evaluation method that can be conveniently pickled."""
    metrics = {}
    l1 = builders.get_loss("l1", cfg)
    metrics["l1"] = l1(y_pred, y)
    return metrics
