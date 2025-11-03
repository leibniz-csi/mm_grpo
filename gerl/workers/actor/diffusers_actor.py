# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn

from gerl import DataProto
from verl.utils.device import get_device_name
from verl.utils.profiler import GPUMemoryLogger
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DiffusersPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DiffusersPPOActor(BasePPOActor):
    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.device_name = get_device_name()

    @GPUMemoryLogger(role="diffusers actor", logger=logger)
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses"""
        # set to eval
        raise NotImplementedError()

    @GPUMemoryLogger(role="diffusers actor", logger=logger)
    def update_policy(self, data: DataProto):
        raise NotImplementedError()
