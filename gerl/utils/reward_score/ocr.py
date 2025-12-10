# Copyright 2025 Huawei Technologies Co., Ltd
#
# Adapted from https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/ocr.py
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

from typing import Union

import numpy as np
import torch
from Levenshtein import distance
from PIL import Image

from .scorer import Scorer


class PaddleOCRScorer(Scorer):
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,  # Disable unnecessary log output
        )

    @torch.no_grad()
    def __call__(
        self,
        images: Union[list[Image.Image], np.ndarray, torch.Tensor],
        prompts: list[str],
    ) -> list[float]:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward scores
        """
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            images = self.array_to_images(images)

        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), (
            "Images and prompts must have the same length"
        )
        for img, prompt in zip(images, prompts):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                # OCR recognition
                result = self.ocr.ocr(img, cls=False)
                # Extract recognized text (handle possible multi-line results)
                recognized_text = (
                    "".join([res[1][0] if res[1][1] > 0 else "" for res in result[0]])
                    if result[0]
                    else ""
                )

                recognized_text = recognized_text.replace(" ", "").lower()
                prompt = prompt.replace(" ", "").lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)

            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            reward = 1 - dist / (len(prompt))
            rewards.append(reward)

        return rewards


def compute_score(images, prompts):
    """
    Compute OCR reward score using PaddleOCR for a batch of images and prompts.
    """
    scorer = PaddleOCRScorer()
    scores = scorer(images, prompts)

    return scores
