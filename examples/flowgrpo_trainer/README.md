# Flow-GRPO: Training Flow Matching Models via Online RL

[Flow-GRPO Paper](https://arxiv.org/abs/2505.05470) | [Original Repo](https://github.com/yifan123/flow_grpo)

*Original Abstract:*
> We propose Flow-GRPO, the first method to integrate online policy gradient reinforcement learning (RL) into flow matching models. Our approach uses two key strategies: (1) an ODE-to-SDE conversion that transforms a deterministic Ordinary Differential Equation (ODE) into an equivalent Stochastic Differential Equation (SDE) that matches the original model's marginal distribution at all timesteps, enabling statistical sampling for RL exploration; and (2) a Denoising Reduction strategy that reduces training denoising steps while retaining the original number of inference steps, significantly improving sampling efficiency without sacrificing performance. Empirically, Flow-GRPO is effective across multiple text-to-image tasks. For compositional generation, RL-tuned SD3.5-M generates nearly perfect object counts, spatial relations, and fine-grained attributes, increasing GenEval accuracy from  to . In visual text rendering, accuracy improves from  to , greatly enhancing text generation. Flow-GRPO also achieves substantial gains in human preference alignment. Notably, very little reward hacking occurred, meaning rewards did not increase at the cost of appreciable image quality or diversity degradation.

## Supported Algorithms
- [x] Flow-GRPO
- [x] Flow-GRPO-Fast

## Quick Start

Below are examples for post-training SD-3.5-M on an OCR task using the OCR reward.

1. Dataset

Download the OCR dataset from [Flow-GRPO](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr) and place it in the `dataset` folder.
<br>
Before training, specify the paths in the configuration parameters `data.train_files` and `data.val_files`.

2. Start Training

We provide scripts for a quick start:

```bash
# SD3 + Flow-GRPO
bash examples/flowgrpo_trainer/run_sd3.sh

# SD3 + Flow-GRPO-Fast
bash examples/flowgrpo_trainer/run_sd3_fast.sh
```

## Performance

> All experiments were conducted on *NVIDIA H800* GPUs using the Paddle OCR reward.

### Training Throughput

| Model   | Algorithm      | # Cards | Batch Size | Learning Rate | Throughput | Script |
| ------- | -------------- | ------  | ---------- | ------------- | ---------- | ------ |
| SD3.5-M | Flow-GRPO      | 8       | 64         | 3e-4          | 0.4        | [run_sd3.sh](./run_sd3.sh) |
| SD3.5-M | Flow-GRPO-Fast | 8       | 64         | 1e-4          | 1.0        | [run_sd3_fast.sh](./run_sd3_fast.sh) |


### Validation Reward Curve

| Model   | Algorithm      | # Cards | Reward Curve |
| ------- | -----------    | ------- | ------------- |
| SD3.5-M | Flow-GRPO      | 8       | <img width=512 src="https://github.com/user-attachments/assets/e559bb07-bca0-4672-b849-f665c5cbc0d1" /> |
| SD3.5-M | Flow-GRPO-Fast | 8       | <img width=512 src="https://github.com/user-attachments/assets/24393445-81e2-43dd-9be9-ec24d85f58dc" /> |
