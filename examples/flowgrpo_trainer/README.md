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
| A100 (80GB) | 530.30.02 | 12.2 |
| H800 (80GB) | 535.161.08 | 12.2 |


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
