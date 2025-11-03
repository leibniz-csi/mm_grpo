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

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs

__all__ = ["DiffusersModelConfig"]


@dataclass
class DiffusersModelConfig(BaseConfig):
    # note that we separate model_path, model_config_path and tokenizer_path in case they are different
    _mutable_fields = {
        "hf_config_path",
        "hf_config",
        "generation_config",
        "local_path",
        "local_hf_config_path",
    }

    path: str = MISSING
    local_path: Optional[str] = None
    hf_config_path: Optional[str] = None
    local_hf_config_path: Optional[str] = None

    hf_config: Any = None
    generation_config: Any = None

    # whether to use shared memory
    use_shm: bool = False

    external_lib: Optional[str] = None

    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False

    # lora related. We may setup a separate config later
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[str] = "all-linear"

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None

    # TODO: in diffusers, these options are no longer used. Drop it later.
    tokenizer_path: Any = None
    trust_remote_code: Any = None
    custom_chat_template: Any = None
    override_config: Any = None
    use_remove_padding: Any = None
    exclude_modules: Any = None
    use_liger: Any = None

    def __post_init__(self):
        import_external_libs(self.external_lib)

        if self.hf_config_path is None:
            self.hf_config_path = self.path

        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)
        if self.hf_config_path != self.path:
            self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
