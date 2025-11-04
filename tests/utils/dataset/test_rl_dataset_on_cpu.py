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

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def get_textprompt_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/dataset/ocr")
    local_path = os.path.join(local_folder, "train.txt")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_rl_dataset():
    from verl.utils.dataset.rl_dataset import collate_fn
    from gerl.utils.dataset.diffusion_dataset import DiffusionTextPromptDataset

    local_path = get_textprompt_data()
    config = OmegaConf.create(
        {
            "train_batch_size": 2,
            "val_max_samples": 8,
            "max_prompt_length": 512,
        }
    )
    dataset = DiffusionTextPromptDataset(data_files=local_path, config=config)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    a = next(iter(dataloader))

    from gerl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    assert "prompt" in data_proto.batch

    output = dataset[0]["prompt"]
    print(f"type: type{output}")
    print(f"\n\noutput: {output}")
