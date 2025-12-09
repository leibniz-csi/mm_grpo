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

import os

import numpy as np
import pytest

from gerl.protocol import DataProto
from gerl.workers.config import DiffusersModelConfig, DiffusionRolloutConfig
from gerl.workers.rollout.diffusers_rollout import DiffusersSyncRollout


@pytest.fixture
def mock_data() -> DataProto:
    test_prompt = "a photo of a cat"
    data = DataProto(non_tensor_batch={"prompt": np.array([test_prompt])})
    return data


class TestDiffusersRollout:
    def setup_class(self):
        model_path = os.environ.get(
            "MODEL_PATH", "stabilityai/stable-diffusion-3.5-medium"
        )
        diffusion_config = DiffusionRolloutConfig(
            with_reward=False, free_cache_engine=False
        )
        model_config = DiffusersModelConfig(path=model_path)
        self.rollout_engine = DiffusersSyncRollout(diffusion_config, model_config, None)

    def test_generate_sequences(self, mock_data: DataProto):
        result = self.rollout_engine.generate_sequences(mock_data)
        expected_batch_keys = [
            "responses",
            "latents",
            "timesteps",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
        for key in expected_batch_keys:
            assert key in result.batch, f"Key {key} not found in result batch."

        assert result.batch.batch_size[0] == 1, (
            f"Expected batch size 1, got {result.batch.batch_size[0]}."
        )
        assert "cached_steps" in result.meta_info, (
            "cached_steps not found in meta_info."
        )

    @pytest.mark.asyncio
    async def test_update_weights(self):
        await self.rollout_engine.update_weights({})

    @pytest.mark.asyncio
    async def test_resume(self):
        await self.rollout_engine.resume()

    @pytest.mark.asyncio
    async def test_release(self):
        await self.rollout_engine.release()
