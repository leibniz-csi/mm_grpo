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
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["FSDPEngineConfig"]


@dataclass
class FSDPEngineConfig(BaseConfig):
    """Configuration for FSDP (Fully Sharded Data Parallel).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        wrap_policy (Dict[str, Any]): Configuration for FSDP wrap policy.
        param_offload (bool): Whether to offload parameters to CPU, default False
        optimizer_offload (bool): Whether to offload optimizer states to CPU, default False
        offload_policy (bool): Whether to offload policy model parameters, default False
        reshard_after_forward (bool): Whether to reshard parameters after forward pass, default True
        fsdp_size (int): FSDP group size. -1 means use all available GPUs.
        forward_prefetch (bool): Whether to prefetch parameters for next forward pass, default False
        model_dtype (str): Model data type used to initialize the transformers model. default "fp32"
        use_orig_params (bool): Whether to use original parameters when initialize FSDP1, default False
        mixed_precision (Optional[dict[str, Any]]): Mixed precision configuration for FSDP, default None
    """

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    param_offload: bool = False
    optimizer_offload: bool = False
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    use_orig_params: bool = False
    mixed_precision: Optional[dict[str, Any]] = None
    strategy: str = "fsdp"

    def __post_init__(self):
        assert self.strategy in ["fsdp", "fsdp2"], (
            f"strategy {self.strategy} not supported"
        )
