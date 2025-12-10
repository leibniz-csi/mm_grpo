# Copyright 2025 Huawei Technologies Co., Ltd
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

import asyncio
import base64
import logging
import os
import re
from io import BytesIO
from typing import Optional, Union

import Levenshtein
import numpy as np
import torch
from openai import AsyncOpenAI
from PIL import Image

from .scorer import Scorer

logger = logging.getLogger(__name__)


class VLLMScorer(Scorer):
    def __init__(self, base_url: Optional[str] = None) -> None:
        # following https://github.com/openai/openai-python/issues/1254
        # we should use a single event loop for AsyncOpenAI call
        self._loop = asyncio.new_event_loop()
        self.aclient = AsyncOpenAI(base_url=base_url, api_key="EMPTY")

    async def async_process_queries(
        self, queries: list[list[dict]], model_path: str, base_url: str
    ) -> list[str]:
        results = await asyncio.gather(
            *(
                self._async_query_openai(query, model_path, base_url)
                for query in queries
            )
        )
        return results

    async def _async_query_openai(
        self, query: list[dict], model_path: str, base_url: str
    ) -> str:
        completion = await self.aclient.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": query,
                },
            ],
            max_completion_tokens=256,
            temperature=0,
        )
        return completion.choices[0].message.content

    @torch.no_grad()
    def __call__(
        self,
        images: Union[list[Image.Image], np.ndarray, torch.Tensor],
        prompts: list[str],
    ) -> list[float]:
        raise NotImplementedError("This method should be implemented in subclasses.")

    @staticmethod
    def pil_image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_image = f"data:image;base64,{encoded_image_text}"
        return base64_image


class QwenVLOCRVLLMScorer(VLLMScorer):
    """
    Calculate OCR reward via calling vllm serving VQA models, e.g. Qwen-VL, UnifiedReward models.
    You can set environment variables `QWEN_VL_OCR_VLLM_URL` and `QWEN_VL_OCR_PATH` to configure the model to use.
    """

    _DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    _task = "Please output only the text content from the image without any additional descriptions or formatting."

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = os.environ.get("QWEN_VL_OCR_VLLM_URL", base_url)
        self.model_path = os.environ.get("QWEN_VL_OCR_PATH", self._DEFAULT_MODEL)
        super().__init__(base_url=self.base_url)

    @torch.no_grad()
    def __call__(
        self,
        images: Union[list[Image.Image], np.ndarray, torch.Tensor],
        prompts: Optional[list[str]] = None,
    ) -> list[float]:
        """
        Calculate OCR reward.

        Args:
            images (list[Image], numpy, tensor):  List of input images
            prompts (list[str]): Corresponding target text list

        Returns:
            list[float]: Reward scores
        """
        assert prompts is not None, "Prompts must be provided for OCR scoring."
        assert self.base_url is not None, (
            "Base URL for OCR VLLM scorer must be provided."
        )

        if isinstance(images, (np.ndarray, torch.Tensor)):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            images = self.array_to_images(images)

        images_base64 = [self.pil_image_to_base64(image) for image in images]
        queries = [self.prepare_query(image_base64) for image_base64 in images_base64]

        results = self._loop.run_until_complete(
            self.async_process_queries(queries, self.model_path, self.base_url)
        )
        logger.debug("VLLM output: %s", results)

        rewards = self.calculate_score(results, prompts)
        return rewards

    def prepare_query(self, image_base64: str) -> list:
        query = [
            {
                "type": "image_url",
                "image_url": {"url": image_base64},
            },
            {"type": "text", "text": self._task},
        ]
        return query

    @staticmethod
    def calculate_score(output_text: list[str], prompts: list[str]) -> list[float]:
        scores = []
        for text, prompt in zip(output_text, prompts):
            # remove any nonvisible characters and convert to lowercase
            prompt = re.sub(r"\s+", "", prompt).lower()
            text = re.sub(r"\s+", "", text).lower()
            dist = Levenshtein.distance(text, prompt)

            # recognized many unrelated characters, only add one character penalty
            dist = min(dist, len(prompt))

            score = 1 - dist / len(prompt)
            scores.append(score)

        return scores


class UnifiedRewardVLLMScorer(VLLMScorer):
    """
    Calculate image scores with captions via calling vllm serving VQA models, e.g. Qwen-VL, UnifiedReward models.
    You can set environment variables `UNIFIED_REWARD_VLLM_URL` and `UNIFIED_REWARD_PATH` to configure the model to use,
    """

    _DEFAULT_MODEL = "UnifiedReward"
    _task = (
        "You are given a text caption and a generated image based on that caption. "
        "Your task is to evaluate this image based on two key criteria:\n"
        "1. Alignment with the Caption: Assess how well this image aligns with the provided caption. "
        "Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
        "2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, "
        "color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, "
        "evaluate their presence in the generated image using the format: "
        "'element (type): value' (where value=0 means not generated, and value=1 means generated), "
        "and assign a score from 1 to 5 after 'Final Score:'.\n"
        "Your task is provided as follows:\n"
        "Text Caption: [{prompt}]"
    )

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = os.environ.get("UNIFIED_REWARD_VLLM_URL", base_url)
        self.model_path = os.environ.get("UNIFIED_REWARD_PATH", self._DEFAULT_MODEL)
        super().__init__(base_url=self.base_url)

    @torch.no_grad()
    def __call__(
        self,
        images: Union[list[Image.Image], np.ndarray, torch.Tensor],
        prompts: Optional[list[str]] = None,
    ) -> list[float]:
        """
        Calculate Image reward based on caption alignment and image quality.

        Args:
            images (list[Image], numpy, tensor):  List of input images
            prompts (list[str]): Corresponding captions

        Returns:
            list[float]: Reward scores
        """
        assert prompts is not None, "Prompts must be provided for scoring."
        assert self.base_url is not None, "Base URL for VLLM scorer must be provided."
        if isinstance(images, (np.ndarray, torch.Tensor)):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            images = self.array_to_images(images)

        images_base64 = [self.pil_image_to_base64(image) for image in images]
        queries = [
            self.prepare_query(image_base64, prompt)
            for image_base64, prompt in zip(images_base64, prompts)
        ]

        results = self._loop.run_until_complete(
            self.async_process_queries(queries, self.model_path, self.base_url)
        )
        logger.debug("VLLM output: %s", results)

        rewards = self.calculate_score(results)
        return rewards

    def prepare_query(self, image_base64: str, prompt: str) -> list:
        query = [
            {
                "type": "image_url",
                "image_url": {"url": image_base64},
            },
            {"type": "text", "text": self._task.format(prompt=prompt)},
        ]
        return query

    @staticmethod
    def calculate_score(output_text: list[str]) -> list[float]:
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in output_text:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)) / 5)
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        scores = [max(0, min(1, score)) for score in scores]
        return scores


def run_qwen_vl_ocr_vllm_scorer():
    scorer = QwenVLOCRVLLMScorer("http://0.0.0.0:9529/v1")
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg", "assets/ocr.jpg"]
    # original prompt: 'a photo of displaying "OCR".'
    prompts = ["OCR"] * len(images)
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images, prompts=prompts))


def run_unified_reward_vllm_scorer():
    scorer = UnifiedRewardVLLMScorer("http://0.0.0.0:8090/v1")
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    prompts = ["a photo of apple."] * len(images)
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images, prompts=prompts))


if __name__ == "__main__":
    run_qwen_vl_ocr_vllm_scorer()
    run_unified_reward_vllm_scorer()
