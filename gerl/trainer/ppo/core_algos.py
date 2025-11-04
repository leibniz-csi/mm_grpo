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
Additional core functions to implement FlowGRPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss
from verl.trainer.ppo.core_algos import AdvantageEstimator as verlAdvantageEstimator


class AdvantageEstimator(verlAdvantageEstimator):
    """Using an enumeration class to avoid spelling errors in adv_estimator.
    """
    FLOW_GRPO = "flow_grpo" # newly added for diffusion models


@register_adv_est(AdvantageEstimator.FLOW_GRPO)
def compute_flow_grpo_outcome_advantage(
    instance_level_rewards: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        instance_level_rewards: `(torch.Tensor)`
            shape is (bs, )
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, )
        Returns: `(torch.Tensor)`
            shape is (bs, )
    """
    scores = instance_level_rewards

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

    return scores, scores

@register_policy_loss("vanilla_diffusion")  # type: ignore[arg-type]
def compute_policy_loss_vanilla_diffusion(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    t_step: int,
    config: Optional[DictConfig | AlgoConfig] = None,
) -> torch.Tensor:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3_fast.py#L885

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    # TODO (Mike): add clip_max to ActorConfig
    clip_max = config.get("clip_max", 5.0)
    clip_ratio = config.clip_ratio

    old_log_prob = old_log_prob[:, t_step]

    advantages = torch.clamp(
        advantages,
        -clip_max,
        clip_max,
    )
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - clip_ratio,
        1.0 + clip_ratio,
    )
    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
    return policy_loss