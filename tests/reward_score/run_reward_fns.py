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

"""
This script demonstrates how to use various reward scorers.
Please refer to testing script `./run_reward_fns.sh` for server setup details.
"""

import asyncio

from PIL import Image

from gerl.utils.reward_score import multi, ocr, vllm


def run_multi_scorer():
    scorers = {"paddle-ocr": 1.0}
    # Instantiate scorer
    scorer = multi.MultiScorer(scorers)
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg", "assets/ocr.jpg"]
    prompts = ["OCR"] * len(images)
    pil_images = [Image.open(img) for img in images]
    # Call scorer and print result
    print(asyncio.run(scorer(images=pil_images, prompts=prompts)))


def run_paddle_ocr_scorer():
    example_image_path = "assets/ocr.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = "OCR"
    scorer = ocr.PaddleOCRScorer(use_gpu=False)
    print(asyncio.run(scorer([example_image], [example_prompt])))


def run_qwen_vl_ocr_vllm_scorer():
    scorer = vllm.QwenVLOCRVLLMScorer("http://0.0.0.0:9529/v1")
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg", "assets/ocr.jpg"]
    prompts = ["OCR"] * len(images)
    pil_images = [Image.open(img) for img in images]
    print(asyncio.run(scorer(images=pil_images, prompts=prompts)))


def run_unified_reward_vllm_scorer():
    scorer = vllm.UnifiedRewardVLLMScorer("http://0.0.0.0:8090/v1")
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    prompts = ["a photo of apple."] * len(images)
    pil_images = [Image.open(img) for img in images]
    print(asyncio.run(scorer(images=pil_images, prompts=prompts)))


if __name__ == "__main__":
    run_multi_scorer()
    run_paddle_ocr_scorer()
    run_qwen_vl_ocr_vllm_scorer()
    run_unified_reward_vllm_scorer()
