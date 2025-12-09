# MM-GRPO
A fast and easy-to-use library to support RL training for multi-modal generative models, built on top of `verl`, `vLLM-Omni`, and `diffusers`.


## Key Features

- Easy integration of diverse RL training algorithms for multi-modal generative models, including `FlowGRPO` and `FlowGRPO-Fast`.
- Scalable and efficient parallel training with an asynchronous streaming workflow.
- Compatibility with diffusion models from `diffusers`.

### Supported Algorithms
- [x] [Flow-GRPO](https://arxiv.org/abs/2505.05470)
- [x] Flow-GRPO-Fast
- [ ] [Mix-GRPO](https://arxiv.org/html/2507.21802v1) (coming soon)
- [ ] [DiffusionNFT](https://arxiv.org/abs/2509.16117) (coming soon)

### Supported Async Strategies
- [x] Async Reward Computation during Rollout
- [x] [One-Step-Off Async Policy](https://arxiv.org/abs/2410.18252)

### Supported Models

- [x] [Stable-Diffusion-3.5](https://arxiv.org/abs/2403.03206)

### Supported Rewards

- [x] [PaddleOCR](https://arxiv.org/abs/2109.03144)
- [x] [Qwen2.5VL-OCR](https://arxiv.org/abs/2502.13923)


*Note: This repository is continuously updated. New models, rewards, and algorithms will be added soon.*


## Get Started

### Installation

**Requirements**

- Install the necessary packages:
  ```bash
  pip install -r requirements.txt
  ```

- Install the `verl` main branch: 
  ```bash
  git clone https://github.com/volcengine/verl.git && cd verl && pip install -e .
  ```

**Environment Setup**

Clone this repository:
```bash
git clone https://github.com/leibniz-csi/mm_grpo.git && cd mm_grpo

# Install other required packages for specific rewards, e.g., for the Paddle-OCR reward:
# pip install paddlepaddle==2.6.2 paddleocr==2.9.1 python-Levenshtein
```


### Quick Start

#### Flow-GRPO / Flow-GRPO-Fast

Below are examples for post-training SD-3.5-M on an OCR task using the OCR reward.

1. Dataset

Download the OCR dataset from [Flow-GRPO](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr) and place it in the `dataset` folder.
<br>
Before training, specify the paths in the configs `data.train_files` and `data.val_files`.

2. Start Training

We provide scripts for a quick start:
```bash
# SD3 + Flow-GRPO
bash examples/flowgrpo_trainer/run_sd3.sh

# SD3 + Flow-GRPO-Fast
bash examples/flowgrpo_trainer/run_sd3_fast.sh
```


## Acknowledgement
We appreciate the contributions of the following works:
- [verl](https://github.com/volcengine/verl)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [diffusers](https://github.com/huggingface/diffusers)
- [Flow-GRPO](https://github.com/yifan123/flow_grpo)
