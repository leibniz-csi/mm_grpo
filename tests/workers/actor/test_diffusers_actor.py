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

import pytest
import torch

from gerl.protocol import DataProto
from gerl.workers.actor.diffusers_actor import DiffusersPPOActor
from gerl.workers.config import DiffusionFSDPActorConfig


@pytest.fixture
def mock_data() -> DataProto:
    batch_size = 2
    cached_steps = 3
    data = DataProto.from_dict(
        {
            "latents": torch.randn(batch_size, cached_steps + 1, 16, 64, 64),
            "timesteps": torch.ones((batch_size, cached_steps)).float() * 1000.0,
            "prompt_embeds": torch.randn(batch_size, 589, 4096),
            "pooled_prompt_embeds": torch.randn(batch_size, 2048),
            "negative_prompt_embeds": torch.randn(batch_size, 589, 4096),
            "negative_pooled_prompt_embeds": torch.randn(batch_size, 2048),
            "advantages": torch.randn(batch_size),
            "old_log_probs": torch.randn(batch_size, cached_steps),
            "ref_prev_sample_mean": torch.randn(
                batch_size, cached_steps + 1, 16, 64, 64
            ),
        }
    )
    data.meta_info["micro_batch_size"] = 1
    data.meta_info["cached_steps"] = cached_steps
    return data


class TestDiffusersActor:
    def setup_class(self):
        from diffusers.models.transformers import SD3Transformer2DModel

        from gerl.workers.diffusers_model.schedulers import (
            FlowMatchSDEDiscreteScheduler,
        )

        model_path = os.environ.get(
            "MODEL_PATH", "stabilityai/stable-diffusion-3.5-medium"
        )
        config = DiffusionFSDPActorConfig()
        actor_module = SD3Transformer2DModel.from_pretrained(
            pretrained_model_name_or_path=model_path,
            subfolder="transformer",
            device_map="cuda",
        )
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_path, subfolder="scheduler"
        )
        optimizer = torch.optim.AdamW(actor_module.parameters(), lr=0.0001)
        self.actor_engine = DiffusersPPOActor(
            config, actor_module, scheduler, optimizer
        )

    def test_compute_log_prob(self, mock_data: DataProto):
        batch_size = mock_data.batch.batch_size[0]
        cached_steps = mock_data.meta_info["cached_steps"]
        log_probs, prev_sample_mean = self.actor_engine.compute_log_prob(mock_data)
        assert log_probs.shape == (batch_size, cached_steps), (
            f"Expected log_probs shape ({batch_size}, {cached_steps}), got {log_probs.shape}."
        )
        assert prev_sample_mean.shape == (batch_size, cached_steps, 16, 64, 64), (
            f"Expected prev_sample_mean shape ({batch_size}, {cached_steps}, 16, 64, 64), got {prev_sample_mean.shape}."
        )

    def test_update_policy(self, mock_data: DataProto):
        metrics = self.actor_engine.update_policy(mock_data)
        assert "actor/pg_loss" in metrics, "Policy gradient loss not found in metrics."
