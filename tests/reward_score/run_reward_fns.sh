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
This script tests different reward scorers
"""
# put images "assets/good.jpg", "assets/fair.jpg", "assets/poor.jpg", "assets/ocr.jpg"

# if use local vllm server:
# CUDA_VISIBLE_DEVICES=0 vllm serve ${CHECKPOINT_HOME}/Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 9529
export QWEN_VL_OCR_VLLM_URL=http://0.0.0.0:9529/v1
export QWEN_VL_OCR_PATH=${CHECKPOINT_HOME}/Qwen/Qwen2.5-VL-7B-Instruct
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ${CHECKPOINT_HOME}/CodeGoat24/UnifiedReward-Think-qwen3vl-8b \
#     --host 0.0.0.0 \
#     --trust-remote-code \
#     --served-model-name UnifiedReward \
#     --gpu-memory-utilization 0.9 \
#     --tensor-parallel-size 4 \
#     --pipeline-parallel-size 1 \
#     --limit-mm-per-prompt.image 32 \
#     --port 8090
export UNIFIED_REWARD_VLLM_URL=http://0.0.0.0:8090/v1
export UNIFIED_REWARD_PATH=UnifiedReward

python -m tests.reward_score.run_reward_fns
