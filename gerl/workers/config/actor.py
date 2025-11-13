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

from dataclasses import dataclass, field
from typing import Literal, Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.config.model import HFModelConfig
from verl.workers.config.optimizer import OptimizerConfig

__all__ = ["DiffusionActorConfig", "DiffusionFSDPActorConfig"]


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'flow_grpo'
    """

    loss_mode: str = "flow_grpo"


@dataclass
class DiffusionActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        use_kl_loss (bool): Whether to use KL divergence loss.
        kl_loss_coef (float): KL divergence loss coefficient.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 8
    clip_ratio: float = 0.00001
    clip_max: float = 5.0
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.04
    ppo_epochs: int = 1
    shuffle: bool = False
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    data_loader_seed = 1
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    guidance_scale: float = 4.5
    noise_level: float = 0.7
    sde_type: Literal["sde", "cps"] = "sde"

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING

    def validate(
        self, n_gpus: int, train_batch_size: int, model_config: Optional[dict] = None
    ):
        """Validate actor configuration with runtime parameters."""
        if train_batch_size < self.ppo_mini_batch_size:
            raise ValueError(
                f"train_batch_size ({train_batch_size}) must be >= "
                f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
            )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(
                f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
            )

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class DiffusionFSDPActorConfig(DiffusionActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()

    def validate(
        self, n_gpus: int, train_batch_size: int, model_config: Optional[dict] = None
    ):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)
