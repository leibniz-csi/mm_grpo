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

from omegaconf import DictConfig
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DiffusionTextPromptDataset(Dataset):
    def __init__(
        self, data_files: str, config: DictConfig, max_samples: int = -1, **kwargs
    ):
        self.file_path = os.path.join(data_files)
        self.max_samples = max_samples
        with open(self.file_path) as f:
            self.prompts = [line.strip() for line in f.readlines()]

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        if self.filter_overlong_prompts:
            self.prompts = [x for x in self.prompts if len(x) <= self.max_prompt_length]

        if self.max_samples > 0 and self.max_samples < len(self.prompts):
            self.prompts = self.prompts[: self.max_samples]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "reward_model": {"style": "rule"},
            "data_source": "ocr",
        }
