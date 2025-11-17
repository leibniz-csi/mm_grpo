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

import logging
import os
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class TextPromptDataset(Dataset):
    def __init__(
        self, data_files: str, config: DictConfig, max_samples: int = -1, **kwargs
    ):
        self.file_path = os.path.join(data_files)
        self.max_samples = max_samples
        with open(self.file_path) as f:
            self.prompts = [line.strip() for line in f.readlines()]

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        if self.truncation == "error":
            for prompt in self.prompts:
                raise RuntimeError(
                    f"Prompt length {len(prompt)} is longer than {self.max_prompt_length}."
                )

        if self.filter_overlong_prompts:
            self.prompts = [x for x in self.prompts if len(x) <= self.max_prompt_length]

        if self.max_samples > 0 and self.max_samples < len(self.prompts):
            self.prompts = self.prompts[: self.max_samples]

        self.data_source = config.data_source
        self.reward_model_style = config.reward_model_style

    @staticmethod
    def get_ground_truth(prompt: str, data_source: str):
        if data_source == "ocr":
            ground_truth = prompt.split('"')[1]
            return ground_truth
        else:
            return None

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        item = {
            "prompt": self.prompts[idx],
            "reward_model": {"style": self.reward_model_style},
            "data_source": self.data_source,
        }
        ground_truth = self.get_ground_truth(item["prompt"], item["data_source"])
        if ground_truth is not None:
            item["reward_model"]["ground_truth"] = ground_truth
        return item
