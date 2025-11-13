# Copyright 2025 Huawei Technologies Co., Ltd
#
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import io
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .scorer import Scorer


def jpeg_incompressibility():
    def _fn(images, prompts=None):
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers, strict=False):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts):
        rew, meta = jpeg_fn(images, prompts)
        return -rew / 500, meta

    return _fn


class JpegImcompressibilityScorer(Scorer):
    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Calculate jpeg imcompressibility reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: unused
        :return: Reward scores
        """
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            images = self.array_to_images(images)
        retval, _ = jpeg_incompressibility()(images, None)
        retval = retval.tolist()
        return retval


def compute_score(images, prompts=None) -> float:
    scorer = JpegImcompressibilityScorer()
    scores = scorer(images, prompts)
    return scores


def test_jpeg_compressibility_scorer():
    scorer = JpegImcompressibilityScorer()
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


if __name__ == "__main__":
    test_jpeg_compressibility_scorer()
