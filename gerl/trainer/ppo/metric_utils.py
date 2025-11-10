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
Metrics related to the FlowGRPO trainer.
"""

from typing import Any

import torch

from gerl import DataProto


def _compute_diffusion_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[
        0
    ]  # BCHW # TODO: To determine the meaning
    prompt_length = torch.tensor(
        [len(prompt) for prompt in batch.non_tensor_batch["prompt"]]
    )

    return dict(
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_diffusion_data_metrics(batch: DataProto) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for FlowGRPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["instance_level_scores"]
    sequence_reward = batch.batch["instance_level_rewards"]

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    response_info = _compute_diffusion_response_info(batch)
    prompt_length = response_info["prompt_length"].float()
    # response_length = response_info["response_length"]

    score_mean = torch.mean(sequence_score).detach().item()
    score_max = torch.max(sequence_score).detach().item()
    score_min = torch.min(sequence_score).detach().item()

    reward_mean = torch.mean(sequence_reward).detach().item()
    reward_max = torch.max(sequence_reward).detach().item()
    reward_min = torch.min(sequence_reward).detach().item()

    valid_adv = advantages
    valid_returns = returns

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics


def compute_diffusion_timing_metrics(
    batch: DataProto, timing_raw: dict[str, float]
) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in FlowGRPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage
    """
    response_info = _compute_diffusion_response_info(batch)
    num_prompt = response_info["prompt_length"].shape[0]

    num_instances_of_section = {
        **{
            name: num_prompt
            for name in ["gen", "reward", "old_log_probs", "adv", "update_actor"]
        },
    }
    # keys: start_profile, generation_sequences, generation_timing/max,min,topk_ratio,
    #       gen, reward, old_log_probs, adv, update_actor, step, stop_profile
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_prompt_ms/{name}": timing_raw[name]
            * 1000
            / num_instances_of_section[name]
            for name in set(num_instances_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_diffusion_throughout_metrics(
    batch: DataProto, timing_raw: dict[str, float], n_gpus: int
) -> dict[str, Any]:
    """
    Computes throughput metrics for FlowGRPO training.

    This function calculates performance metrics related to generation processing speed,
    including the total number of images processed, time per step, and throughput
    (images per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about image counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    response_info = _compute_diffusion_response_info(batch)
    total_num_instances = response_info["prompt_length"].shape[0]
    time = timing_raw["step"]
    # TODO
    #  estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,

    return {
        "perf/total_num_tokens": total_num_instances,
        "perf/time_per_step": time,
        "perf/throughput": total_num_instances / (time * n_gpus),
    }
