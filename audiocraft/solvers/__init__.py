# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Solvers. A Solver is a training recipe, combining the dataloaders, models,
optimizer, losses etc into a single convenient object.
"""

# flake8: noqa
from .audiogen import AudioGenSolver
from .base import StandardSolver
from .builders import get_solver
from .compression import CompressionSolver
from .compression_fMRI import CompressionfMRISolver
from .CSM_raw_fMRI import CSMrawfMRI
from .diffusion import DiffusionSolver
from .downstream_CSM_raw_fMRI import DownstreamCSMrawfMRI
from .downstream_fMRI import DownstreamfMRISolver
from .musicgen import MusicGenSolver
