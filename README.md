# MM-GRPO

An easy-to-use and fast library to support RL training for multi-modal generative models, built on top of verl, vLLM, and diffusers.

## Key Features

- Easy-to-integrate diverse RL training algorithms for MM generative models, including `FlowGRPO` and `FlowGRPO-Fast`.
- Scalable and efficient parallel training with asynchronous streaming workflow
- Compatible with diffusion models from `diffusers`.

### Supported Algorithms

- [x] [Flow-GRPO](https://arxiv.org/abs/2505.05470)
- [x] Flow-GRPO-Fast
- [ ] [Mix-GRPO](https://arxiv.org/html/2507.21802v1) (coming soon)
- [ ] [DiffusionNFT](https://arxiv.org/abs/2509.16117) (coming soon)

### Supported models

- [x] [Stable-Diffusion-3.5](https://arxiv.org/abs/2403.03206)

### Supported Rewards

- [x] [PaddleOCR](https://arxiv.org/abs/2109.03144)
- [x] [Qwen2.5VL-OCR](https://arxiv.org/abs/2502.13923)
- [~] LLM As a Judge Rubric with an arbitrary OpenAI Client (Work in progress)

_Note: This repository is continuously updated. New models, rewards, and algorithms will be added soon._

## Get Started

### Installation

**Requirements**

- install necessary packages first by
  ```bash
  pip install -r requirements.txt
  ```
- install `verl` main branch by
  ```bash
  git clone https://github.com/volcengine/verl.git && cd verl && pip install -e .
  ```

**Environment Setup**

Clone this repository:

```bash
git clone https://github.com/leibniz-csi/mm_grpo.git && cd mm_grpo

# install other required packages for specific rewards, e.g., for Paddle-OCR reward
# pip install paddlepaddle==2.6.2 paddleocr==2.9.1 python-Levenshtein
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

## Acknowledgement

We appreciate the contribution of following works:

- [verl](https://github.com/volcengine/verl)
- [Flow-GRPO](https://github.com/yifan123/flow_grpo)
