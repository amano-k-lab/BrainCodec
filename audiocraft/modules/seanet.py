# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import torch
import torch.nn as nn

from .conv import StreamableConv1d, StreamableConvTranspose1d
from .lstm import StreamableLSTM


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "Number of kernel sizes should match number of dilations"
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamableConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = StreamableConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        lstm: int = 0,
        disable_norm_outer_blocks: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            StreamableConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = "none" if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=block_norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            StreamableConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks == self.n_blocks else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        lstm: int = 0,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm="none" if self.disable_norm_outer_blocks == self.n_blocks else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = "none" if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamableConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=block_norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y


class SEANetfMRIEncoder(nn.Module):
    def __init__(
        self,
        space_dim: int = 1024,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        updownsample_kernel_size: int = 2,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        time_extension_ratios: tp.List[int] = [1, 1, 1, 1],
        lstm: int = 0,
        bidiractional: bool = False,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
        tr_emb_dim: int = 256,
        max_tr: float = 256.0,
        tr_precision: float = 0.2,
        add_noise: bool = False,
        noise_lstm: bool = False,
        noise_type: str = "gaussian",
    ):
        super().__init__()
        self.space_dim = space_dim
        self.dimension = dimension
        self.n_filters = n_filters
        # reverse していることに注意
        self.ratios = list(reversed(ratios))
        self.time_extension_ratios = list(reversed(time_extension_ratios))
        assert len(ratios) == len(
            time_extension_ratios
        ), "Number of ratios should match number of time extension ratios"
        assert all(
            [time_extension_ratio >= 1 for time_extension_ratio in time_extension_ratios]
        ), "Time extension ratios should be greater or equal to 1"
        del ratios, time_extension_ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        model: tp.List[nn.Module] = [
            StreamableConv1d(
                space_dim,
                n_filters,
                kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        # Downsample to raw audio scale
        current_filter = n_filters + 0
        for i, (ratio, time_extension_ratio) in enumerate(zip(self.ratios, self.time_extension_ratios)):
            next_filter = int(current_filter * ratio)

            block_norm = "none" if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        current_filter,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=block_norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            # Add downsampling layers
            if time_extension_ratio > 1:
                model += [
                    act(**activation_params),
                    StreamableConvTranspose1d(
                        current_filter,
                        next_filter,
                        kernel_size=time_extension_ratio * 2,
                        stride=time_extension_ratio,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        trim_right_ratio=trim_right_ratio,
                    ),
                ]
            else:
                model += [
                    act(**activation_params),
                    StreamableConv1d(
                        current_filter,
                        next_filter,
                        kernel_size=updownsample_kernel_size,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                    ),
                ]
            current_filter = next_filter + 0

        if lstm:
            model += [StreamableLSTM(current_filter, num_layers=lstm, bidirectional=bidiractional)]

        model += [
            act(**activation_params),
            StreamableConv1d(
                current_filter if bidiractional is False else current_filter * 2,
                dimension if add_noise is False else dimension * 2,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks == self.n_blocks else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)
        self.tr_emb_dim = tr_emb_dim
        self.tr_emb_pad_id = int(max_tr / tr_precision) + 1
        self.emb_linear = nn.Linear(self.tr_emb_dim, self.space_dim)

        self.add_noise = add_noise
        self.noise_lstm = noise_lstm
        self.noise_type = noise_type  # Used for NoisyEncodecfMRIModel
        assert noise_type in ["gaussian", "uniform"]
        self.bidiractional = bidiractional

    def forward(self, x, tr):
        if tr is not None:
            tr = self.emb_linear(tr)
            return self.model(x + tr.transpose(1, 2))
        else:
            return self.model(x)


class SEANetfMRIDecoder(nn.Module):
    def __init__(
        self,
        space_dim: int = 1024,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        updownsample_kernel_size: int = 2,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        time_extension_ratios: tp.List[int] = [1, 1, 1, 1],
        lstm: int = 0,
        bidiractional: bool = False,
        disable_norm_outer_blocks: int = 0,
        use_transpose: bool = True,
        trim_right_ratio: float = 1.0,
        tr_emb_dim: int = 256,
        max_tr: float = 256.0,
        tr_precision: float = 0.2,
        use_inverse_ratios: bool = False,
        add_noise: bool = False,
        noise_lstm: bool = False,
        noise_type: str = "gaussian",
    ):
        super().__init__()
        self.dimension = dimension
        self.space_dim = space_dim
        self.n_filters = n_filters
        self.ratios = ratios
        self.time_extension_ratios = time_extension_ratios
        self.prod_time_extension_ratios = np.prod(time_extension_ratios)
        del ratios, time_extension_ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)

        current_filter = n_filters + 0
        for ratio in self.ratios:
            current_filter = int(current_filter * ratio)

        model: tp.List[nn.Module] = [
            StreamableConv1d(
                dimension,
                current_filter,
                kernel_size,
                norm="none" if self.disable_norm_outer_blocks == self.n_blocks else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [
                StreamableLSTM(
                    current_filter,
                    num_layers=lstm,
                    bidirectional=bidiractional,
                )
            ]
            n_filters = n_filters * 2 if bidiractional is True else n_filters

        # Upsample to raw audio scale
        if use_inverse_ratios is True:
            self.ratios = list(reversed(self.ratios))
        for i, (ratio, time_extension_ratio) in enumerate(zip(self.ratios, self.time_extension_ratios)):
            next_filter = int(current_filter / ratio)

            block_norm = "none" if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            # Add upsampling layers
            if use_transpose is True:
                if time_extension_ratio > 1:
                    raise ValueError("Transpose convolution does not support time extension ratio > 1")
                model += [
                    act(**activation_params),
                    StreamableConvTranspose1d(
                        current_filter,
                        next_filter,
                        kernel_size=updownsample_kernel_size,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        trim_right_ratio=trim_right_ratio,
                    ),
                ]
            else:
                if time_extension_ratio > 1:
                    model += [
                        act(**activation_params),
                        StreamableConv1d(
                            current_filter,
                            next_filter,
                            kernel_size=time_extension_ratio * 2,
                            stride=time_extension_ratio,
                            norm=block_norm,
                            norm_kwargs=norm_params,
                            causal=causal,
                            pad_mode=pad_mode,
                        ),
                    ]
                else:
                    model += [
                        act(**activation_params),
                        StreamableConv1d(
                            current_filter,
                            next_filter,
                            kernel_size=updownsample_kernel_size,
                            norm=block_norm,
                            norm_kwargs=norm_params,
                            causal=causal,
                            pad_mode=pad_mode,
                        ),
                    ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        next_filter,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=block_norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            current_filter = next_filter + 0

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv1d(
                current_filter,
                space_dim,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)
        self.tr_emb_dim = tr_emb_dim
        self.tr_emb_pad_id = int(max_tr / tr_precision) + 1
        self.emb_linear = nn.Linear(self.tr_emb_dim, self.dimension)

    def forward(self, z, tr, mu=None, logvar=None):
        if mu is not None:  # for noise_disable = True
            noise = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
            if z is None:
                z = noise
            else:
                z = z + noise

        # Create time embedding like [1,2] → [1,1,2,2]
        if tr is not None:
            B, emb_dim = tr.size(0), tr.size(2)
            tr = tr.unsqueeze(-2).repeat(1, 1, self.prod_time_extension_ratios, 1).view(B, -1, emb_dim)
            tr = self.emb_linear(tr)
            y = self.model(z + tr.transpose(1, 2))
        else:
            y = self.model(z)
        return y


class SEANetfMRIPRVQEncoder(nn.Module):
    def __init__(
        self,
        space_dim: int = 1024,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        updownsample_kernel_size: int = 2,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        time_extension_ratios: tp.List[int] = [1, 1, 1, 1],
        lstm: int = 0,
        bidiractional: bool = False,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
        tr_emb_dim: int = 256,
        max_tr: float = 256.0,
        tr_precision: float = 0.2,
        q_logvar: bool = False,
    ):
        super().__init__()
        self.space_dim = space_dim
        self.dimension = dimension
        self.n_filters = n_filters
        # reverse していることに注意
        self.ratios = list(reversed(ratios))
        self.time_extension_ratios = list(reversed(time_extension_ratios))
        assert len(ratios) == len(
            time_extension_ratios
        ), "Number of ratios should match number of time extension ratios"
        assert all(
            [time_extension_ratio >= 1 for time_extension_ratio in time_extension_ratios]
        ), "Time extension ratios should be greater or equal to 1"
        del ratios, time_extension_ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        model: tp.List[nn.Module] = [
            StreamableConv1d(
                space_dim,
                n_filters,
                kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        # Downsample to raw audio scale
        current_filter = n_filters + 0
        for i, (ratio, time_extension_ratio) in enumerate(zip(self.ratios, self.time_extension_ratios)):
            next_filter = int(current_filter * ratio)

            block_norm = "none" if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        current_filter,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=block_norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            # Add downsampling layers
            if time_extension_ratio > 1:
                model += [
                    act(**activation_params),
                    StreamableConvTranspose1d(
                        current_filter,
                        next_filter,
                        kernel_size=time_extension_ratio * 2,
                        stride=time_extension_ratio,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        trim_right_ratio=trim_right_ratio,
                    ),
                ]
            else:
                model += [
                    act(**activation_params),
                    StreamableConv1d(
                        current_filter,
                        next_filter,
                        kernel_size=updownsample_kernel_size,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                    ),
                ]
            current_filter = next_filter + 0

        if lstm:
            model += [StreamableLSTM(current_filter, num_layers=lstm, bidirectional=bidiractional)]

        model += [
            act(**activation_params),
            StreamableConv1d(
                current_filter if bidiractional is False else current_filter * 2,
                dimension * 2,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks == self.n_blocks else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)
        self.tr_emb_dim = tr_emb_dim
        self.tr_emb_pad_id = int(max_tr / tr_precision) + 1
        self.emb_linear = nn.Linear(self.tr_emb_dim, self.space_dim)

        self.bidiractional = bidiractional

        self.q_logvar = q_logvar

    def forward(self, x, tr):
        tr = self.emb_linear(tr)
        return self.model(x + tr.transpose(1, 2))


class SEANetfMRIPRVQDecoder(nn.Module):
    def __init__(
        self,
        space_dim: int = 1024,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        updownsample_kernel_size: int = 2,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        time_extension_ratios: tp.List[int] = [1, 1, 1, 1],
        lstm: int = 0,
        bidiractional: bool = False,
        disable_norm_outer_blocks: int = 0,
        use_transpose: bool = True,
        trim_right_ratio: float = 1.0,
        tr_emb_dim: int = 256,
        max_tr: float = 256.0,
        tr_precision: float = 0.2,
        use_inverse_ratios: bool = False,
        q_logvar: bool = False,
    ):
        super().__init__()
        self.dimension = dimension
        self.space_dim = space_dim
        self.n_filters = n_filters
        self.ratios = ratios
        self.time_extension_ratios = time_extension_ratios
        self.prod_time_extension_ratios = np.prod(time_extension_ratios)
        del ratios, time_extension_ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)

        current_filter = n_filters + 0
        for ratio in self.ratios:
            current_filter = int(current_filter * ratio)

        model: tp.List[nn.Module] = [
            StreamableConv1d(
                dimension,
                current_filter,
                kernel_size,
                norm="none" if self.disable_norm_outer_blocks == self.n_blocks else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [
                StreamableLSTM(
                    current_filter,
                    num_layers=lstm,
                    bidirectional=bidiractional,
                )
            ]
            n_filters = n_filters * 2 if bidiractional is True else n_filters

        # Upsample to raw audio scale
        if use_inverse_ratios is True:
            self.ratios = list(reversed(self.ratios))
        for i, (ratio, time_extension_ratio) in enumerate(zip(self.ratios, self.time_extension_ratios)):
            next_filter = int(current_filter / ratio)

            block_norm = "none" if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1) else norm
            # Add upsampling layers
            if use_transpose is True:
                if time_extension_ratio > 1:
                    raise ValueError("Transpose convolution does not support time extension ratio > 1")
                model += [
                    act(**activation_params),
                    StreamableConvTranspose1d(
                        current_filter,
                        next_filter,
                        kernel_size=updownsample_kernel_size,
                        norm=block_norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        trim_right_ratio=trim_right_ratio,
                    ),
                ]
            else:
                if time_extension_ratio > 1:
                    model += [
                        act(**activation_params),
                        StreamableConv1d(
                            current_filter,
                            next_filter,
                            kernel_size=time_extension_ratio * 2,
                            stride=time_extension_ratio,
                            norm=block_norm,
                            norm_kwargs=norm_params,
                            causal=causal,
                            pad_mode=pad_mode,
                        ),
                    ]
                else:
                    model += [
                        act(**activation_params),
                        StreamableConv1d(
                            current_filter,
                            next_filter,
                            kernel_size=updownsample_kernel_size,
                            norm=block_norm,
                            norm_kwargs=norm_params,
                            causal=causal,
                            pad_mode=pad_mode,
                        ),
                    ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        next_filter,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=block_norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            current_filter = next_filter + 0

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv1d(
                current_filter,
                space_dim,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)
        self.tr_emb_dim = tr_emb_dim
        self.tr_emb_pad_id = int(max_tr / tr_precision) + 1
        self.emb_linear = nn.Linear(self.tr_emb_dim, self.dimension)

    def forward(self, mu, logvar, tr):
        z = mu
        if logvar is not None:  # for noise_disable = True
            noise = torch.randn_like(z) * torch.exp(0.5 * logvar)
            z = z + noise

        # Create time embedding like [1,2] → [1,1,2,2]
        B, emb_dim = tr.size(0), tr.size(2)
        tr = tr.unsqueeze(-2).repeat(1, 1, self.prod_time_extension_ratios, 1).view(B, -1, emb_dim)
        tr = self.emb_linear(tr)
        y = self.model(z + tr.transpose(1, 2))
        return y
