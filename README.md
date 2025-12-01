# MM-GRPO (gerl)

An easy-to-use and fast library to support RL training for multi-modal generative models, built on top of verl, vLLM, and diffusers.

> **Package Name**: `gerl` - Generative RL for multi-modal models

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

_Note: This repository is continuously updated. New models, rewards, and algorithms will be added soon._

## Get Started

### Installation

**Requirements**

- Python 3.10 or higher (Python 3.12 recommended)
- [uv](https://docs.astral.sh/uv/) package manager

**Environment Setup**

1. Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone this repository:

```bash
git clone https://github.com/leibniz-csi/mm_grpo.git && cd mm_grpo
```

3. Install the package and dependencies:

```bash
# Install with default dependency groups (includes wandb)
uv sync

# Or install with additional dependency groups
uv sync --group dev  # for development tools
uv sync --group ocr  # for PaddleOCR support
uv sync --all-groups # install all dependency groups
```

The package will automatically install `verl` from the upstream Git repository as configured in `pyproject.toml`.

### Quick Start

**Flow-GRPO / Flow-GRPO-Fast**

Below we provide examples to post-train SD-3.5-M on OCR task using OCR reward.

1. **Dataset**

Download OCR dataset from [Flow-GRPO](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr) and place it under `dataset` folder.

During training, denote paths in configs `data.train_files` and `data.val_files`.

2. **Start Training**

The package provides a command-line interface via the `gerl_flowgrpo` script:

```bash
# Activate the uv environment
source .venv/bin/activate  # or use: uv run

# SD3 + Flow-GRPO
bash examples/flowgrpo_trainer/run_sd3.sh

# SD3 + Flow-GRPO-Fast
bash examples/flowgrpo_trainer/run_sd3_fast.sh
```

Alternatively, you can run commands directly with uv:

```bash
# Run without activating the virtual environment
uv run gerl_flowgrpo [args]

# Or use the example scripts directly
uv run bash examples/flowgrpo_trainer/run_sd3.sh
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Acknowledgement

We appreciate the contribution of following works:

- [verl](https://github.com/volcengine/verl)
- [Flow-GRPO](https://github.com/yifan123/flow_grpo)
