# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""RVQ."""
from .base import BaseQuantizer, DummyQuantizer, QuantizedResult
# flake8: noqa
from .vq import ResidualVectorQuantizer
