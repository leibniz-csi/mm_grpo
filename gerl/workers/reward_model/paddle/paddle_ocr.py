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
from typing import Union

import numpy as np
import torch
from PIL import Image
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.profiler import DistProfiler, DistProfilerExtension

from ....protocol import DataProto

logger = logging.getLogger(__file__)


class PaddleOCRRewardModelWorker(Worker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)
        self.config = config
        self._register_dispatch_collect_info(
            "reward", dp_rank=self.rank, is_collect=True
        )

    def _build_model(self, config):
        from paddleocr import PaddleOCR

        reward_module = PaddleOCR(
            use_angle_cls=False, lang="en", use_gpu=False, show_log=False
        )
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.reward_module = self._build_model(config=self.config)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        instance_level_scores = self.compute_ocr_score(
            data.batch["responses"], data.non_tensor_batch["prompt"]
        )
        output = DataProto.from_dict(
            tensors={"rm_scores": torch.tensor(instance_level_scores)}
        )
        return output

    @staticmethod
    def array_to_images(images: Union[np.ndarray, torch.Tensor]) -> list[Image.Image]:
        if isinstance(images, torch.Tensor):
            images = images.float().permute(0, 2, 3, 1).cpu().numpy()
        assert images.shape[-1] == 3, "must be in NHWC format"
        images = (images * 255).round().clip(0, 255).astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
        return images

    def compute_ocr_score(
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
        from Levenshtein import distance

        if isinstance(images, (np.ndarray, torch.Tensor)):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            images = self.array_to_images(images)

        scores = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), (
            "Images and prompts must have the same length"
        )
        for img, prompt in zip(images, prompts):
            prompt = prompt.split('"')[1]

            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                # OCR recognition
                result = self.reward_module.ocr(img, cls=False)
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
                logger.warning(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            score = 1 - dist / (len(prompt))
            scores.append(score)

        return scores
