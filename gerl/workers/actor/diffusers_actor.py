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
Single Process Actor
"""

import logging
import os
import random
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from gerl import DataProto
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DiffusersPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DiffusersPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        pipeline: DiffusionPipeline,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.role = "Ref" if actor_optimizer is None else "Actor"
        self.scheduler = pipeline.scheduler
        self.device_name = get_device_name()

        # TODO: add these to config
        self.noise_level = config.get("noise_level", 0.7)
        self.sde_type = config.get("sde_type", "sde")
        self.guidance_scale = config.get("guidance_scale", 7.0)

    def _forward_micro_batch(self, micro_batch) -> tuple[torch.Tensor, int]:
        """
        Returns:
            log_probs: # (bs, )
        """

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            latents = micro_batch["latents"]
            timesteps = micro_batch["timesteps"]
            prompt_embeds = micro_batch["prompt_embeds"]
            pooled_prompt_embeds = micro_batch["pooled_prompt_embeds"]
            negative_prompt_embeds = micro_batch["negative_prompt_embeds"]
            negative_pooled_prompt_embeds = micro_batch["negative_pooled_prompt_embeds"]
            # TODO (Mike): instead of computing all log_probs, we randomly sample one timestep to compute log_prob
            # better do in this way: shuffle the data in the outside of the loop, and then we can compute
            # log_probs for all timesteps
            t_step = random.randint(0, timesteps.shape[1] - 1)
            if self.guidance_scale > 1.0:
                noise_pred = self.actor_module(
                    hidden_states=torch.cat([latents[:, t_step]] * 2),
                    timestep=torch.cat([timesteps[:, t_step]] * 2),
                    encoder_hidden_states=torch.cat(
                        [negative_prompt_embeds, prompt_embeds], dim=0
                    ),
                    pooled_projections=torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                    ),
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = self.actor_module(
                    hidden_states=latents[:, t_step],
                    timestep=timesteps[:, t_step],
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

            _, log_probs, _, _ = self.scheduler.compute_log_prob(
                sample=latents[:, t_step],
                model_output=noise_pred.float(),
                timestep=timesteps[:, t_step],
                noise_level=self.noise_level,
                prev_sample=latents[:, t_step + 1].float(),
                sde_type=self.sde_type,
            )

        return log_probs, t_step

    def _optimizer_step(self):
        assert self.config.grad_clip is not None and self.actor_optimizer is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(
                max_norm=self.config.grad_clip
            )
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(
                f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}"
            )
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="diffusers actor", logger=logger)
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses"""
        # for flow-grpo, we do not need to recompute log probs
        return data.batch["log_probs"]

    @GPUMemoryLogger(role="diffusers actor", logger=logger)
    def update_policy(self, data: DataProto):
        assert self.actor_optimizer is not None

        # make sure we are in training mode
        self.actor_module.train()

        select_keys = [
            "latents",
            "old_log_probs",
            "advantages",
            "timesteps",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
        ]
        if self.config.use_kl_loss:
            raise NotImplementedError(
                "KL loss is not implemented for DiffusersPPOActor yet."
            )

        non_tensor_select_keys: list = []

        data = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics: dict = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size
                    // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches = mini_batch.split(
                    self.config.ppo_micro_batch_size_per_gpu
                )

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    loss_scale_factor = 1 / self.gradient_accumulation

                    log_prob, t_step = self._forward_micro_batch(model_inputs)

                    # for fully_async_policy recipe
                    if (
                        hasattr(self.config, "use_rollout_log_probs")
                        and self.config.use_rollout_log_probs
                    ):
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get(
                        "loss_mode", "vanilla_diffusion"
                    )

                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (all functions return 4 values)
                    policy_loss = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        t_step=t_step,
                        config=self.config,
                    )

                    if self.config.use_kl_loss:
                        raise NotImplementedError

                    loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": policy_loss.detach().item()
                            * loss_scale_factor,
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
