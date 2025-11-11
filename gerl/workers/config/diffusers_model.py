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
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs

__all__ = ["DiffusersModelConfig"]


@dataclass
class DiffusersModelConfig(BaseConfig):
    _mutable_fields = {"local_path"}

    path: str = MISSING
    local_path: Optional[str] = None

    # whether to use shared memory
    use_shm: bool = False

    external_lib: Optional[str] = None

    enable_gradient_checkpointing: bool = False
    enable_activation_offload: bool = False

    # lora related. We may setup a separate config later
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_init_weights: str = "gaussian"
    target_modules: Optional[str] = "auto"
    exclude_modules: Optional[str] = None

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None

    use_fused_kernels: bool = False

    def __post_init__(self):
        import_external_libs(self.external_lib)
        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)
