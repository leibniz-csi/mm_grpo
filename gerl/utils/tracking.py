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

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: Optional[str] = None
    experiment_name: Optional[str] = None

    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        wandb.log(
            {
                "val/generations": [
                    wandb.Image(
                        image.float(),
                        caption=f"Prompt: {prompt}\n Score: {score}",
                        file_type="jpg",
                    )
                    for prompt, image, score in samples
                ]
            },
            step=step,
        )
