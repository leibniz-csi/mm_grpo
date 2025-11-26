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


def prepare_train_network(
    pipeline: "DiffusionPipeline",
    device: torch.device,
    dtype: torch.dtype,
    is_lora: bool = True,
) -> None:
    """Prepare the training network by moving it to the specified device and dtype.

    Args:
        pipeline (DiffusionPipeline): The diffusion pipeline.
        device (torch.device): The target device.
        dtype (torch.dtype): The target data type.
        is_lora (bool): Whether to use LoRA for training.
    """
    from diffusers import StableDiffusion3Pipeline

    if isinstance(pipeline, StableDiffusion3Pipeline):
        # freeze parameters of models to save more memory
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
        pipeline.text_encoder_3.requires_grad_(False)
        pipeline.transformer.requires_grad_(not is_lora)
        # Move vae and text_encoder to device and cast to inference_dtype
        pipeline.vae.to(device)
        pipeline.text_encoder.to(device, dtype=dtype)
        pipeline.text_encoder_2.to(device, dtype=dtype)
        pipeline.text_encoder_3.to(device, dtype=dtype)
        # set eval mode
        pipeline.vae.eval()
        pipeline.text_encoder.eval()
        pipeline.text_encoder_2.eval()
        pipeline.text_encoder_3.eval()
    else:
        raise NotImplementedError()
