# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class StreamableLSTM(nn.Module):
    """LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        bidirectional: bool = False,
        skip: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional=bidirectional)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            if self.bidirectional is True:
                x = torch.cat((x, x), dim=-1)
            y = y + x
        y = y.permute(1, 2, 0)
        return y


class StreamableLSTMforTimeCompression(nn.Module):
    """LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional=bidirectional)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        _, (hs, _) = self.lstm(x)
        return torch.mean(hs, dim=0)


if __name__ == "__main__":
    lstm = StreamableLSTMforTimeCompression(1024, 2, bidirectional=True)
    dummpy_input = torch.rand(16, 1024, 64)

    output = lstm(dummpy_input)
    print(output.shape)
