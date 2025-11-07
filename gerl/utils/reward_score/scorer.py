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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image


class Scorer(ABC):
    @abstractmethod
    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> Union[List[float], Dict[str, List[float]]]:
        """Return the scoring value of the images"""
        pass

    @staticmethod
    def array_to_images(images: Union[np.ndarray, torch.Tensor]) -> List[Image.Image]:
        if isinstance(images, torch.Tensor):
            images = images.permute(0, 2, 3, 1).cpu().numpy()
        assert images.shape[-1] == 3, "must be in NHWC format"
        images = (images * 255).round().clip(0, 255).astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
        return images
