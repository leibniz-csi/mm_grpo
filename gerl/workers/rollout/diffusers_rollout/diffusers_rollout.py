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
"""
Rollout with diffusers models.
"""

import logging
import os
from typing import Generator

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh

from gerl import DataProto
from verl.utils.device import get_device_name
from verl.utils.profiler import GPUMemoryLogger
from gerl.workers.config import DiffusersModelConfig
from verl.workers.config import RolloutConfig
from verl.workers.rollout.base import BaseRollout

__all__ = ["DiffusersRollout"]


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DiffusersRollout(BaseRollout):
    def __init__(
        self,
        rollout_module: DiffusionPipeline,
        config: RolloutConfig,
        model_config: DiffusersModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.rollout_module = rollout_module
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

    @GPUMemoryLogger(role="diffusers rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # TODO (Mike): hard coded, for test only, make is configurable later
        HEIGHT, WIDTH = 512, 512
        input_texts = prompts.non_tensor_batch["prompt"].tolist()

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.rollout_module(
                input_texts,
                height=HEIGHT,
                width=WIDTH,
                max_sequence_length=self.config.prompt_length,
                output_type="pt",
            )

        # TODO (Mike): hard coded, for test only, drop later
        images_pil = output.images.cpu().float().permute(0, 2, 3, 1).numpy()
        images_pil = (images_pil * 255).round().astype("uint8")
        os.makedirs("visual", exist_ok=True)
        for image in images_pil:
            assert image.shape == (HEIGHT, WIDTH, 3)
            uuid = os.urandom(8).hex()
            Image.fromarray(image).save(f"visual/{uuid}.jpg")

        batch = TensorDict(
            {
                "responses": output.images,
                "latents": output.all_latents,
                "log_probs": output.all_log_probs,
                "timesteps": output.all_timesteps,
                "prompt_embeds": output.prompt_embeds,
                "pooled_prompt_embeds": output.pooled_prompt_embeds,
                "negative_prompt_embeds": output.negative_prompt_embeds,
                "negative_pooled_prompt_embeds": output.negative_pooled_prompt_embeds,
            },
            batch_size=len(output.images),
        )

        return DataProto(batch=batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        pass

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

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        pass
