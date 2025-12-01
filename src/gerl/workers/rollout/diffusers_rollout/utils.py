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
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline


def get_negative_prompt_embedding(
    pipeline: "DiffusionPipeline",
    negative_prompt: str = "",
    max_sequence_length: int = 128,
) -> dict[str, torch.FloatTensor]:
    from diffusers import StableDiffusion3Pipeline

    if isinstance(pipeline, StableDiffusion3Pipeline):
        prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            negative_prompt,
            None,
            None,
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )
        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_pooled_prompt_embeds": pooled_prompt_embeds,
        }
    else:
        raise NotImplementedError(f"Pipeline type {type(pipeline)} not supported yet.")
