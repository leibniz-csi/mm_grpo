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

import importlib
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from .scorer import Scorer

AVAILABLE_SCORERS = {
    "paddle-ocr": ("ocr", "PaddleOCRScorer"),
    # "jpeg-imcompressibility": ("jpeg_imcompressibility", "JpegImcompressibilityScorer"),
    "qwenvl-ocr-vllm": ("vllm", "QwenVLOCRVLLMScorer"),
    "unified-reward-vllm": ("vllm", "UnifiedRewardVLLMScorer"),
    # "pickscore": ("pickscore", "PickScoreScorer"),
    # "qwenvl": ("qwenvl", "QwenVLScorer"),
    # "aesthetic": ("aesthetic", "AestheticScorer"),
    # "jpeg-compressibility": ("compression", "JpegCompressibilityScorer"),
    # "qwenvl-vllm": ("vllm", "QwenVLVLLMScorer"),
}


class MultiScorer(Scorer):
    def __init__(self, scorers: Dict[str, float]) -> None:
        self.score_fn: dict[str, Callable] = dict()
        self.scorers = scorers
        self.init_scorer_cls()

    def init_scorer_cls(self):
        for score_name in self.scorers.keys():
            module, cls = AVAILABLE_SCORERS[score_name]
            module = "gerl.utils.reward_score." + module
            module = importlib.import_module(module)
            cls = getattr(module, cls)
            self.score_fn[score_name] = cls()

    def __call__(
        self,
        images: Union[list[Image.Image], np.ndarray, torch.Tensor],
        prompts: Optional[list[str]] = None,
    ) -> Dict[str, list[float]]:
        """
        Calculate reward scores from multiples scorers
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward scores, including individual scorer results and the total score
        """
        score_details = dict()
        total_scores: list[float] = list()
        for score_name, weight in self.scorers.items():
            scores = self.score_fn[score_name](images, prompts=prompts)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [
                    total + weighted
                    for total, weighted in zip(total_scores, weighted_scores)
                ]

        score_details["score"] = total_scores
        return score_details


def compute_score(images, prompts, scorers: Dict[str, float]):
    """
    Compute multiple reward scores for a batch of images and prompts.
    """
    scorer = MultiScorer(scorers)
    scores = scorer(images, prompts)

    return scores


def run_multi_scorer():
    scorers = {"jpeg-imcompressibility": 1.0}
    scorer = MultiScorer(scorers)
    images = ["assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg"]
    pil_images = [Image.open(img) for img in images]
    print(scorer(images=pil_images))


if __name__ == "__main__":
    run_multi_scorer()
