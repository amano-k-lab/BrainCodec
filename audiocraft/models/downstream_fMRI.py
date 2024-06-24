import logging
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

logger = logging.getLogger()


class DownstreamfMRIModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def space_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def segment_duration(self) -> int:
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @staticmethod
    def get_pretrained(name: str, device: tp.Union[torch.device, str] = "cpu") -> "CompressionModel":
        """Instantiate a CompressionModel from a given pretrained model.

        Args:
            name (Path or str): name of the pretrained model. See after.
            device (torch.device or str): Device on which the model is loaded.

        Pretrained models:
            - dac_44khz (https://github.com/descriptinc/descript-audio-codec)
            - dac_24khz (same)
            - facebook/encodec_24khz (https://huggingface.co/facebook/encodec_24khz)
            - facebook/encodec_32khz (https://huggingface.co/facebook/encodec_32khz)
            - your own model on HugginFace. Export instructions to come...
        """

        from . import builders

        model: DownstreamfMRIModel
        if Path(name).exists():
            pkg = torch.load(name, map_location="cpu")
            cfg = OmegaConf.create(pkg["xp.cfg"])
            cfg.device = str(device)
            model = builders.get_downstream_fMRI_model(cfg)
            model.load_state_dict(pkg["best_state"])
        else:
            raise ValueError(f"Unknown pretrained model {name}")
        return model.to(device).eval()


class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(
            dim=self.dim,
        )


class LinearBaseline(DownstreamfMRIModel):
    space_dim: int = 0
    segment_duration: int = 0
    num_classes: int = 0

    def __init__(
        self,
        space_dim: int,
        segment_duration: int,
        num_classes: int,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.segment_duration = segment_duration
        self.num_classes = num_classes

        model: tp.List[nn.Module] = []

        model += [
            nn.Linear(in_features=self.segment_duration, out_features=1),
            Squeeze(dim=-1),
            nn.Linear(in_features=self.space_dim, out_features=self.num_classes),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LinearBaselineforHidden(DownstreamfMRIModel):
    space_dim: int = 0
    segment_duration: int = 0
    num_classes: int = 0

    def __init__(
        self,
        space_dim: int,
        segment_duration: int,
        num_classes: int,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.segment_duration = segment_duration
        self.num_classes = num_classes

        self.time_linear = nn.Linear(in_features=self.segment_duration, out_features=1)
        self.squeeze = Squeeze(dim=-1)
        self.space_linear = nn.Linear(in_features=self.space_dim, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_linear(x)
        x = self.squeeze(x)
        x = self.space_linear(x)
        return x

    def get_hidden_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_linear(x)
        x = self.squeeze(x)
        return x
