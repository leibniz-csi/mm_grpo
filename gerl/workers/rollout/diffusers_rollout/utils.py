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

from diffusers import DiffusionPipeline, StableDiffusion3Pipeline

from .pipelines import StableDiffusion3PipelineWithLogProb
from .schedulers import FlowMatchSDEDiscreteScheduler


def inject_SDE_scheduler_into_pipeline(pipeline: DiffusionPipeline, pretrained_model_name_or_path: str):
    if isinstance(pipeline, StableDiffusion3Pipeline):
        # override __call__ method
        type(pipeline).__call__ = StableDiffusion3PipelineWithLogProb.__call__
        # replace scheduler
        scheduler_config = FlowMatchSDEDiscreteScheduler.load_config(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        pipeline.scheduler = FlowMatchSDEDiscreteScheduler.from_config(scheduler_config)
    else:
        raise NotImplementedError()
