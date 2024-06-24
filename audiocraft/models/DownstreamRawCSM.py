import typing as tp

import torch
import torch.nn.functional as F

from ..solvers import CompressionfMRISolver, DownstreamCSMrawfMRI
from ..utils.autocast import TorchAutocast
from .downstream_rawlm import DownstreamRawLMModel
from .encodec import CompressionModel


class DownstreamRawCSM:
    def __init__(
        self,
        name: str,
        compression_model: CompressionModel,
        lm: DownstreamRawLMModel,
    ):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        # Just to be safe, let's put everything in eval mode.
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

        lm = DownstreamCSMrawfMRI.model_from_checkpoint(name, device=device)
        compression_model = CompressionfMRISolver.model_from_checkpoint(
            lm.cfg.compression_model_checkpoint, device=device
        )

        return DownstreamRawCSM(name, compression_model, lm)

    def get_prediction(
        self,
        bold: torch.Tensor,  # [B, dim, T]
        bold_durations: tp.List[int],  # [B,]
        tr: tp.List[float],  # [B, ]
    ):
        bold, bold_durations = self.lm.add_start_emb(bold, bold_durations)
        tr_tensor = self.lm.tr_preprocess(tr, bold.shape[-1], bold_durations).to(bold.device)

        bold, tr_tensor = self.lm.add_cls_emb(bold, tr_tensor)

        with self.autocast:
            output = self.lm(bold, tr_tensor, [])
            output = F.softmax(self.lm.num_class_linear(output[:, -1, :]), dim=-1)  # output only last time
        return output
