# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import torch

from ..solvers import CompressionfMRISolver, CSMrawfMRI
from ..utils.autocast import TorchAutocast
from .encodec import CompressionModel
from .rawlm import RawLMModel

logger = logging.getLogger(__name__)


class RawCSM:
    def __init__(
        self,
        name: str,
        compression_model: CompressionModel,
        lm: RawLMModel,
    ):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        # Just to be safe, let's put everything in eval mode.
        if compression_model is not None:
            self.compression_model.eval()
        self.lm.eval()

        self.device = next(iter(lm.parameters())).device
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
        if self.device.type == "cpu":
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(enabled=True, device_type=self.device.type, dtype=torch.float16)

    @staticmethod
    def get_pretrained(name: str, device=None):
        # TODO: これは多分 ckpt からと export したモデルとで分けるべき
        if device is None:
            if torch.cuda.device_count():
                device = "cuda"
            else:
                device = "cpu"

        lm = CSMrawfMRI.model_from_checkpoint(name, device=device)
        if lm.cfg.compression_model_checkpoint is not None:
            compression_model = CompressionfMRISolver.model_from_checkpoint(
                lm.cfg.compression_model_checkpoint, device=device
            )
        else:
            compression_model = None

        return RawCSM(name, compression_model, lm)
