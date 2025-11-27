# Flow-GRPO: Training Flow Matching Models via Online RL

[Flow-GRPO Paper](https://arxiv.org/abs/2505.05470) | [Original Repo](https://github.com/yifan123/flow_grpo)

*Original Abstract:*
> We propose Flow-GRPO, the first method to integrate online policy gradient reinforcement learning (RL) into flow matching models. Our approach uses two key strategies: (1) an ODE-to-SDE conversion that transforms a deterministic Ordinary Differential Equation (ODE) into an equivalent Stochastic Differential Equation (SDE) that matches the original model's marginal distribution at all timesteps, enabling statistical sampling for RL exploration; and (2) a Denoising Reduction strategy that reduces training denoising steps while retaining the original number of inference steps, significantly improving sampling efficiency without sacrificing performance. Empirically, Flow-GRPO is effective across multiple text-to-image tasks. For compositional generation, RL-tuned SD3.5-M generates nearly perfect object counts, spatial relations, and fine-grained attributes, increasing GenEval accuracy from  to . In visual text rendering, accuracy improves from  to , greatly enhancing text generation. Flow-GRPO also achieves substantial gains in human preference alignment. Notably, very little reward hacking occurred, meaning rewards did not increase at the cost of appreciable image quality or diversity degradation.

## Supported Algorithms
- [x] Flow-GRPO
- [x] Flow-GRPO-Fast


## Get Started

### Installation

**Requirements**

We tested with the following machines:
| GPU | Driver | CUDA |
|--- | --- | --- |
| A100 (80GB) | 530.30.02 | 12.1 |
| H800 (80GB) | 535.161.08 | 12.2 |

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

We provide scripts for quick start:
```bash
# SD3 + Flow-GRPO
bash examples/flowgrpo_trainer/run_sd3.sh

# SD3 + Flow-GRPO-Fast
bash examples/flowgrpo_trainer/run_sd3_fast.sh
```

<details>
<summary>Multi-card training</summary>


Example of running on 8 GPUs (at least 60GB memory/card suggested) with Flow-GRPO-Fast and LoRA:
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

Example of running on a single GPU (60GB memory suggested) with Flow-GRPO-Fast and LoRA:
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
    trainer.total_epochs=3 $@
```
</details>

## Performance

- All experiments were conducted under NVIDIA H800 with memory 80GB/card, with Paddle OCR reward.

| model | RL alg. |  cards | batch size | init lr | clip ratio | s/step |
| --- | --- | --- | --- | --- | --- | --- |
| SD3.5-M | Flow-GRPO | 1 | 8 | 3e-4 | 1e-4 | 84 |
| SD3.5-M | Flow-GRPO-Fast | 1 | 8 | 3e-4 | 1e-5 | 58 |
| SD3.5-M | Flow-GRPO-Fast | 8 | 64 | 3e-4 | 1e-5 | 291 |

- Validation reward curve:


| model | RL alg. |  cards  | curve |
| --- | --- | --- | --- |
| SD3.5-M | Flow-GRPO | 1 | <img width=512 alt="val_reward_curve" src="https://github.com/user-attachments/assets/f8c3e0f7-ebc0-4b3a-b25b-c7e349feb69f" />|
| SD3.5-M | Flow-GRPO-Fast | 1 | <img width=512 alt="val_reward_curve" src="https://github.com/user-attachments/assets/4289e040-a3a0-48d8-b2c3-a8d08f27baa4" />|
| SD3.5-M | Flow-GRPO-Fast | 8 | <img width=512 src="https://github.com/user-attachments/assets/db1b84be-d258-441c-88af-8e86d82aa2a8" /> |


- Some visualization comparison for Flow-GRPO-Fast:

| model | RL alg. |  cards | prompt | rendering (before RL)| rendering (after RL)
| --- | --- | --- | --- | --- | --- |
| SD3.5-M | Flow-GRPO | 1  | `A close-up of a sleek smartwatch on a wrist, the screen displaying "Step Goal Achieved" with a celebratory animation, set against a blurred cityscape at dusk, capturing the moment of accomplishment.` | Step 0: <img width=400 src="https://github.com/user-attachments/assets/d6b1cf69-b0d1-4ecb-92c5-c32fc4a8676c" />|  Step 141: <img width=400 src="https://github.com/user-attachments/assets/e4208ff7-2e49-4767-9578-320f12e4d4dd" />|
| SD3.5-M | Flow-GRPO-Fast | 1  | `A high-fashion runway with a sleek, modern backdrop displaying "Spring Collection 2024". Models walk confidently on the catwalk, showcasing vibrant, floral prints and pastel tones, under soft, ambient lighting that enhances the fresh, spring vibe.`|Step 0: <img src="https://github.com/user-attachments/assets/7cf01b78-b310-4473-9ab7-22f5eff97565" width=400> | Step 104:  <img width=400 src="https://github.com/user-attachments/assets/6b02ef1c-3da7-44cd-9850-9114635dea4f" />|
| SD3.5-M | Flow-GRPO-Fast | 8  | `A beautifully crafted birthday cake topper shaped like "30 Years Young", adorned with sparkly frosting and shimmering decorations, set against a backdrop of a cozy, candlelit birthday party.`| Step 0: <img width=400 src="https://github.com/user-attachments/assets/d1e5b85f-4b8c-4645-94e9-8841b64de8e7" />| Step 97: <img width=400 src="https://github.com/user-attachments/assets/51a15d07-6990-420a-a509-0e756b790e40" /> |
