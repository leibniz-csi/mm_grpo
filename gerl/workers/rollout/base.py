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

import importlib
from abc import ABC, abstractmethod
from typing import Generator

import torch
from torch.distributed.device_mesh import DeviceMesh
from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass

from ...workers.config import DiffusersModelConfig, DiffusionRolloutConfig

__all__ = ["BaseRollout"]


class BaseRollout(ABC):
    """Base class for rollout."""

    def __init__(
        self,
        config: DiffusionRolloutConfig,
        model_config: DiffusersModelConfig,
        device_mesh: DeviceMesh,
    ):
        self.config: DiffusionRolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: DiffusersModelConfig = omega_conf_to_dataclass(model_config)
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self):
        """Resume rollout weights in GPU memory."""
        pass

    @abstractmethod
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        pass

    @abstractmethod
    async def release(self):
        """Release weights and kv cache in GPU memory."""
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Args:
            prompts: The input prompts.

        Returns:
            The output sequences.
        """
        raise NotImplementedError


_ROLLOUT_REGISTRY = {
    ("diffusers", "sync"): "gerl.workers.rollout.DiffusersSyncRollout",
    ("diffusers", "async"): "gerl.workers.rollout.DiffusersAsyncRollout",
}


def get_rollout_class(rollout_name: str, mode: str) -> type[BaseRollout]:
    """Get the rollout class by name.

    Args:
        rollout_name: The name of the rollout.
        mode: The mode of the rollout, sync: spmd mode, async: server mode.

    Returns:
        The rollout class.
    """
    assert (rollout_name, mode) in _ROLLOUT_REGISTRY, (
        f"Rollout {rollout_name} with mode {mode} not found"
    )
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]
    module_name, class_name = fqdn.rsplit(".", 1)
    rollout_module = importlib.import_module(module_name)
    return getattr(rollout_module, class_name)
