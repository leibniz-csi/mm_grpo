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

from collections import defaultdict
from typing import Any

import torch
from verl.workers.reward_manager.abstract import AbstractRewardManager

from ...protocol import DataProto
from ...utils.reward_score import DefaultScorer
from .registry import register


@register("diffusion")
class DiffusionRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        reward_fn=[],
    ) -> None:
        """
        Initialize the DiffusionRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function/class to compute the reward score. If None, `DefaultScorer()` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            reward_fn: The name list of reward functions to compute the reward.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or DefaultScorer()
        self.reward_fn_key = (
            reward_fn_key  # Store the key for accessing the data source
        )
        self.reward_fn = reward_fn

    def __call__(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # TODO: not supported yet.
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch[key] for key in reward_extra_keys
                }
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros(len(data.batch["responses"]), dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_str = data_item.non_tensor_batch["prompt"]
            response_images = data_item.batch["responses"]

            ground_truth = data_item.non_tensor_batch["reward_model"].get(
                "ground_truth", prompt_str
            )
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            extra_info["reward_fn"] = self.reward_fn

            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_images,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"][0]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_images.shape)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


@register("diffusion-batch")
class DiffusionBatchRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        reward_fn=[],
    ) -> None:
        """
        A batch reward manager that computes rewards for a batch of data.

        Args:
            tokenizer: unused.
            num_examine (int): The number of responses to examine.
            compute_score (callable): The function to compute the rewards. If None, `default_compute_score` will be used.
            reward_fn_key (str): The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            reward_fn (list): The name list of reward functions to compute the reward.
            reward_kwargs (dict): The keyword arguments to pass to the reward function.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or DefaultScorer()
        self.reward_fn_key = reward_fn_key
        self.reward_fn = reward_fn

    def __call__(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # TODO: not supported yet.
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {
                    key: data.non_tensor_batch[key] for key in reward_extra_keys
                }
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return data.batch["rm_scores"]

        reward_extra_info = defaultdict(list)

        # get batch reward scores
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        prompts = data.non_tensor_batch["prompt"]  # print use
        response_images = data.batch["responses"]
        ground_truths = [
            item.non_tensor_batch["reward_model"].get(
                "ground_truth", item.non_tensor_batch["prompt"]
            )
            for item in data
        ]
        rollout_reward_scores = data.non_tensor_batch.get(
            "reward_scores", [{} for _ in range(len(data))]
        )  # useless for now
        extras = data.non_tensor_batch.get(
            "extra_info", [{} for _ in range(len(data))]
        )  # useless for now
        for i in range(len(data)):
            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]
            extras[i]["reward_fn"] = self.reward_fn

        scores = self.compute_score(
            data_source=data_sources[0],
            solution_str=response_images,
            ground_truth=ground_truths,
            extra_info=extras[0],
        )
        if isinstance(scores, dict):
            rewards = scores["score"]
            for key, value in scores.items():
                reward_extra_info[key] = value
        else:  # list
            rewards = scores
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)

        # print information
        already_printed: dict[str, Any] = {}
        for i in range(len(data)):
            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                print("[prompt]", prompts[i])
                print("[response]", response_images[i].shape)
                print("[ground_truth]", ground_truths[i])
                if isinstance(scores, dict):
                    for key, value in scores.items():
                        print(f"[{key}]", value[i])
                else:
                    print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(
            rewards, dtype=torch.float32, device=response_images.device
        )
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
