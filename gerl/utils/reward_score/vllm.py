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
from typing import Any, List, Optional, Union

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
        self, queries: List[list[Any]], model_path: str, base_url: Optional[str]
    ) -> List[str]:
        results = await asyncio.gather(
            *(
                self._async_query_openai(query, model_path, base_url)
                for query in queries
            )
        )
        return results

    async def _async_query_openai(
        self, query: List[list[Any]], model_path: str, base_url: Optional[str]
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
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: List[str],
    ) -> List[float]:
        raise NotImplementedError("This method should be implemented in subclasses.")


class QwenVLOcrVLLMScorer(VLLMScorer):
    _DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    _task = "Please output only the text content from the image without any additional descriptions or formatting."

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = os.environ.get("QWEN_VL_OCR_VLLM_URL", base_url)
        self.model_path = os.environ.get("QWEN_VL_OCR_PATH", self._DEFAULT_MODEL)
        super().__init__(base_url=self.base_url)

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: List[str],
    ) -> List[float]:
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

    def prepare_query(self, image_base64: str) -> List:
        query = [
            {
                "type": "image_url",
                "image_url": {"url": image_base64},
            },
            {"type": "text", "text": self._task},
        ]
        return query

    @staticmethod
    def pil_image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    @staticmethod
    def calculate_score(output_text: List[str], prompts: List[str]) -> List[float]:
        scores = []
        # assume the prompt is in the format: xxx display/show with "words" xxx
        prompts = [
            prompt.split('"')[1] if prompt.find('"') >= 0 else prompt
            for prompt in prompts
        ]
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


def test_qwen_vl_ocr_vllm_scorer():
    scorer = QwenVLOcrVLLMScorer("http://0.0.0.0:9529/v1")
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg", "assets/ocr.jpg"]
    prompts = ['a photo of displaying "OCR".'] * len(images)
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images, prompts=prompts))


if __name__ == "__main__":
    test_qwen_vl_ocr_vllm_scorer()
