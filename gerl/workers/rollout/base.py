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

from verl.workers.rollout.base import BaseRollout

_ROLLOUT_REGISTRY = {
    ("diffusers", "sync"): "gerl.workers.rollout.diffusers_rollout.DiffusersRollout",
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
