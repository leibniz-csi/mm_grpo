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
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.workers.actor import BasePPOActor

from ...protocol import DataProto
from ...trainer.ppo.core_algos import get_policy_loss_fn, kl_penalty
from ..config import DiffusionActorConfig

__all__ = ["DiffusersPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DiffusersPPOActor(BasePPOActor):
    def __init__(
        self,
        config: DiffusionActorConfig,
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
        self.dtype = (
            torch.float16
            if self.config.fsdp_config.model_dtype == "fp16"
            else torch.bfloat16
        )

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], step: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs: # (bs, )
        """

        with torch.autocast(device_type=self.device_name, dtype=self.dtype):
            latents = micro_batch["latents"]
            timesteps = micro_batch["timesteps"]
            prompt_embeds = micro_batch["prompt_embeds"]
            pooled_prompt_embeds = micro_batch["pooled_prompt_embeds"]
            negative_prompt_embeds = micro_batch["negative_prompt_embeds"]
            negative_pooled_prompt_embeds = micro_batch["negative_pooled_prompt_embeds"]

            if self.config.guidance_scale > 1.0:
                noise_pred = self.actor_module(
                    hidden_states=torch.cat([latents[:, step]] * 2),
                    timestep=torch.cat([timesteps[:, step]] * 2),
                    encoder_hidden_states=torch.cat(
                        [negative_prompt_embeds, prompt_embeds], dim=0
                    ),
                    pooled_projections=torch.cat(
                        [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                    ),
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = self.actor_module(
                    hidden_states=latents[:, step],
                    timestep=timesteps[:, step],
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

            # TODO (Mike): double check if the computation is correct
            prev_sample, log_prob, prev_sample_mean, std_dev_t = (
                self.scheduler.sample_previous_step(
                    sample=latents[:, step],
                    model_output=noise_pred.float(),
                    timestep=timesteps[:, step],
                    noise_level=self.config.noise_level,
                    prev_sample=latents[:, step + 1].float(),
                    sde_type=self.config.sde_type,
                )
            )

        return prev_sample, log_prob, prev_sample_mean, std_dev_t

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        assert self.actor_optimizer is not None

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
        raise NotImplementedError()

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

        non_tensor_select_keys: list = []

        data = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )
        # shuffle samples along batch dimension
        data.reorder(torch.randperm(len(data)))

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics: dict = {}
        train_step = 0
        self.actor_optimizer.zero_grad()

        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps = (
            len(mini_batches) * data.meta_info["cached_steps"] // 2
        )
        loss_scale_factor = 1 / gradient_accumulation_steps

        for _ in range(self.config.ppo_epochs):
            for micro_batch in mini_batches:
                for step in range(micro_batch.meta_info["cached_steps"]):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    _, log_prob, prev_sample_mean, std_dev_t = (
                        self._forward_micro_batch(model_inputs, step=step)
                    )

                    if self.config.use_kl_loss:
                        with torch.no_grad():
                            _, _, prev_sample_mean_ref, _ = self._forward_micro_batch(
                                model_inputs, step=step
                            )

                    policy_loss_fn = get_policy_loss_fn(
                        self.config.policy_loss.loss_mode
                    )

                    # Compute policy loss (all functions return 4 values)
                    policy_loss = policy_loss_fn(
                        old_log_prob=old_log_prob[:, step],
                        log_prob=log_prob,
                        advantages=advantages,
                        config=self.config,
                    )

                    if self.config.use_kl_loss:
                        kld = kl_penalty(
                            prev_sample_mean, prev_sample_mean_ref, std_dev_t
                        )
                        kl_loss = kld.mean()

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = (
                            kl_loss.detach().item() * loss_scale_factor
                        )
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {"actor/pg_loss": policy_loss.detach().item()}
                    )
                    append_to_dict(metrics, micro_batch_metrics)
                    train_step += 1

                    if train_step % gradient_accumulation_steps == 0:
                        grad_norm = self._optimizer_step()
                        self.actor_optimizer.zero_grad()
                        mini_batch_metrics = {
                            "actor/grad_norm": grad_norm.detach().item()
                        }
                        append_to_dict(metrics, mini_batch_metrics)

        return metrics
