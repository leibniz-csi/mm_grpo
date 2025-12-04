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
from verl.single_controller.ray import RayWorkerGroup


def update_weights(actor_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup):
    """Update weights from actor worker group to rollout worker group."""
    if actor_wg is rollout_wg:
        return

    per_tensor_param, peft_config = actor_wg.get_params()
    rollout_wg.update_weights(per_tensor_param, peft_config=peft_config)
