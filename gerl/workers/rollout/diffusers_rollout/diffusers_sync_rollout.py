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
from typing import TYPE_CHECKING, Generator, Optional

import numpy as np
import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from verl.utils.device import get_device_name
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.rollout.base import BaseRollout

from ....protocol import DataProto
from ...config import DiffusersModelConfig, DiffusionRolloutConfig

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

__all__ = ["DiffusersSyncRollout"]


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DiffusersSyncRollout(BaseRollout):
    def __init__(
        self,
        config: DiffusionRolloutConfig,
        model_config: DiffusersModelConfig,
        device_mesh: DeviceMesh,
        rollout_module: Optional["DiffusionPipeline"] = None,
    ):
        super().__init__(config, model_config, device_mesh)
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh
        if rollout_module is None:
            raise ValueError("rollout_module must be provided for DiffusersSyncRollout")
        self.rollout_module = rollout_module
        self.dtype = PrecisionType.to_dtype(config.dtype)

        self._cached_prompt_embeds: Optional[dict[str, torch.Tensor]] = None

    @GPUMemoryLogger(role="diffusers rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if self._cached_prompt_embeds is None:
            self._cached_prompt_embeds = self.cache_prompt_embeds()
        negative_prompt_embeds = self._cached_prompt_embeds[
            "negative_prompt_embeds"
        ].to(get_device_name())
        negative_pooled_prompt_embeds = self._cached_prompt_embeds[
            "negative_pooled_prompt_embeds"
        ].to(get_device_name())

        self.rollout_module.transformer.eval()
        micro_batches = prompts.split(self.config.micro_batch_size_per_gpu)
        generated_input_texts = []
        generated_results = []

        seed = prompts.meta_info.get("seed", None)
        if seed is not None:
            generator = torch.Generator(device=get_device_name()).manual_seed(seed)
        else:
            generator = None

        for micro_batch in micro_batches:
            input_texts = micro_batch.non_tensor_batch["prompt"].tolist()

            noise_level = micro_batch.meta_info.get(
                "noise_level", self.config.noise_level
            )
            num_inference_steps = micro_batch.meta_info.get(
                "num_inference_steps", self.config.num_inference_steps
            )

            with torch.autocast(device_type=get_device_name(), dtype=self.dtype):
                output = self.rollout_module(
                    input_texts,
                    height=self.config.image_height,
                    width=self.config.image_width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    generator=generator,
                    max_sequence_length=self.config.prompt_length,
                    negative_prompt_embeds=negative_prompt_embeds.repeat(
                        len(input_texts), 1, 1
                    ),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.repeat(
                        len(input_texts), 1
                    ),
                    output_type="pt",
                    noise_level=noise_level,
                    sde_window_size=self.config.sde_window_size,
                    sde_window_range=self.config.sde_window_range,
                    sde_type=self.config.sde_type,
                )

            result = TensorDict(
                {
                    "responses": output.images,
                    "latents": output.all_latents,
                    "old_log_probs": output.all_log_probs,
                    "timesteps": output.all_timesteps,
                    "prompt_embeds": output.prompt_embeds,
                    "pooled_prompt_embeds": output.pooled_prompt_embeds,
                    "negative_prompt_embeds": output.negative_prompt_embeds,
                    "negative_pooled_prompt_embeds": output.negative_pooled_prompt_embeds,
                },
                batch_size=len(output.images),
            )

            generated_results.append(result)
            generated_input_texts.extend(input_texts)

        result = DataProto(
            batch=torch.cat(generated_results),
            non_tensor_batch={"prompt": np.array(generated_input_texts)},
        )
        result.meta_info["cached_steps"] = result.batch["timesteps"].shape[1]
        return result

    def cache_prompt_embeds(self, negative_prompt: str = "") -> torch.Tensor:
        # TODO (Mike): for stable diffusion 3 only now, need to generalize later
        prompt_embeds, _, pooled_prompt_embeds, _ = self.rollout_module.encode_prompt(
            negative_prompt,
            negative_prompt,
            negative_prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=self.config.prompt_length,
        )
        return {
            "negative_prompt_embeds": prompt_embeds.to("cpu"),
            "negative_pooled_prompt_embeds": pooled_prompt_embeds.to("cpu"),
        }

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
