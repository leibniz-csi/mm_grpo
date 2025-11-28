# Copyright 2025 Huawei Technologies Co., Ltd
#
# Modified from https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl.base_config import BaseConfig

__all__ = ["AlgoConfig"]


@dataclass
class RolloutCorrectionConfig(BaseConfig):
    bypass_mode: bool = False


@dataclass
class AlgoConfig(BaseConfig):
    """Configuration for the algorithm.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        adv_estimator (str): Advantage estimator type: "flow_grpo".
        norm_adv_by_std_in_grpo (bool): Whether to normalize advantages by std.
        global_std (bool): Whether to use global standard deviation for advantage normalization.
    """

    adv_estimator: str = "flow_grpo"
    norm_adv_by_std_in_grpo: bool = True
    global_std: bool = True
    rollout_correction: Optional[RolloutCorrectionConfig] = None
