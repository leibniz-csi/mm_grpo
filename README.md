# MM-GRPO
An easy-to-use and fast library to support RL training for multi-modal generative models, built on top of verl, vLLM, and diffusers.


## Key Features

- Easy-to-integrate diverse RL training algorithms for MM generative models, including FlowGRPO and MixedGRPO
- Scalable and efficient parallel training with asynchronous streaming workflow
- Compatible with diffusion models from `Diffusers`.


## Get Started

### Installation

**Requirements**

Install required packages:
- torch, datasets, diffusers, transformers, peft, flashinfer-python
- [verl](https://verl.readthedocs.io/en/latest/start/install.html) (>0.6.1)
  <br>Note: we support FSDP/FSDP2, and do not support either vllm or sglang for now.

For other optional requirements, please refer to `requirements.txt`

**Environment Setup**

Clone this repository:
```bash
git clone https://github.com/leibniz-csi/mm_grpo.git
cd mm_grpo

# install other required packages for specific rewards, e.g., for Paddle-OCR reward
# pip install paddlepaddle-gpu==2.6.2 paddleocr==2.9.1 python-Levenshtein
```

**Model Download**

Models
- SD3.5: `stabilityai/stable-diffusion-3.5-medium`

Reward Models:

- PaddleOCR:
  ```bash
    # install related packages
    pip install paddlepaddle-gpu==2.6.2 paddleocr==2.9.1 python-Levenshtein
  ```
  ```python
  # pre-download model by running the Python script:
  from paddleocr import PaddleOCR
  ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
  ```


### Quick Start

**Flow-GRPO / Flow-GRPO-Fast**

Below we provide examples to post-train SD-3.5-M on OCR task using OCR reward.

1. Dataset

Download OCR dataset from [Flow-GRPO](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr) and place it under `dataset` folder.
<br>
During training, denote paths in configs `data.train_files` and `data.val_files`.

2. Start Training

<details open>
<summary>Multi-card training</summary>

We provide scripts for quick start:
```bash
# sd3 + Flow-GRPO
bash examples/flowgrpo_trainer/run_sd3.sh

# sd3 + Flow-GRPO-Fast
bash examples/flowgrpo_trainer/run_sd3_fast.sh
```


Example of running on 8 GPUs with Flow-GRPO-Fast:
```bash
python3 -m gerl.trainer.main_flowgrpo \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$HOME/dataset/ocr/train.txt \
    data.val_files=$HOME/dataset/ocr/test.txt \
    data.train_batch_size=64 \
    data.val_max_samples=64 \
    data.max_prompt_length=128 \
    data.filter_overlong_prompts=False \
    data.data_source=ocr \
    data.reward_fn='["paddle-ocr"]' \
    actor_rollout_ref.model.path=stabilityai/stable-diffusion-3.5-medium \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.clip_ratio=1e-5 \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.name=diffusers \
    actor_rollout_ref.rollout.n=24 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.guidance_scale=1.0 \
    actor_rollout_ref.rollout.noise_level=0.8 \
    actor_rollout_ref.rollout.sde_type="cps" \
    actor_rollout_ref.rollout.sde_window_size=3 \
    actor_rollout_ref.rollout.sde_window_range="[0,5]" \
    reward_model.reward_manager=diffusion-batch \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='flow_grpo' \
    trainer.experiment_name='sd35_m_ocr_fast' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 $@
```

</details>

<details>
<summary>Single-card training</summary>

Example of running on a single GPU (60GB memory suggested) with Flow-GRPO-Fast:
```bash
python3 -m gerl.trainer.main_flowgrpo \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$HOME/dataset/ocr/train.txt \
    data.val_files=$HOME/dataset/ocr/test.txt \
    data.train_batch_size=8 \
    data.val_max_samples=16 \
    data.max_prompt_length=128 \
    data.filter_overlong_prompts=False \
    data.data_source=ocr \
    data.reward_fn='["paddle-ocr"]' \
    actor_rollout_ref.model.path=stabilityai/stable-diffusion-3.5-medium \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.clip_ratio=1e-5 \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.name=diffusers \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.guidance_scale=1.0 \
    actor_rollout_ref.rollout.noise_level=0.8 \
    actor_rollout_ref.rollout.sde_type="cps" \
    actor_rollout_ref.rollout.sde_window_size=3 \
    actor_rollout_ref.rollout.sde_window_range="[0,5]" \
    reward_model.reward_manager=diffusion-batch \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='flow_grpo' \
    trainer.experiment_name='sd35_m_ocr_fast' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@
```
</details>


## Supported Rewards

We support multiple rewards by listing reward function names in config `data.reward_fn`, e.g.:
```json
[
    "jpeg-imcompressibility",
    "paddle-ocr",
]
```
The final reward is the (equal) weighted sum of all rewards.

Supported rewards:
- "jpeg-imcompressibility": measures image size as a proxy for quality.
- "paddle-ocr": Paddle-OCR model based OCR reward.
- "qwenvl-ocr-vllm": Qwen-VL model (called via vllm API) based OCR reward.


## Acknowledgement
We appreciate the contribution of following works:
- [verl](https://github.com/volcengine/verl)
- [Flow-GRPO](https://github.com/yifan123/flow_grpo)