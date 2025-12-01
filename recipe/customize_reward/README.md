# Customize Reward Function

`gerl` supports multiple reward scorers in a diversity of rewards, including rule-based, model-based (CPU), and remote API-calling model-based rewards.
Example code is located in `gerl/utils/reward_score`.

Here below are step-by-step instructions on customizing a new reward scorer and some examples.

## Customizing Reward Scorer in Steps

**Step 1: add scorer class in format.**

You can add a new reward scorer in `gerl/utils/reward_score`, for example, by adding scorer script `new_reward.py` in the format below:

```python
from .scorer import Scorer

class CustomizedRewardScorer(Scorer):
    """
    The new scorer inherits `Scorer` class from `gerl/utils/reward_score/score.py`.
    """
    def __init__(self):
        """
        Necessary initialization
        """

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Calculate customized reward here.
        Currently it only supports reward computation between `generated images` and `ground-truth prompts`.

        Inputs:
            - images: generated images
            - prompts: ground-truths (if applicable)
        Return:
            - rewards: the output reward is a list of values for each pair of image and prompt.
        """
        # compute rewards
        return rewards
```

**Step 2: register your reward scorer.**

Add new scorer name and mapped class name in `gerl/utils/reward_score/multi.py`
```python
AVAILABLE_SCORERS = {
    ...
    # "reward_fn_name_in_config": ("scorer_script_location", "scorer_class_name")
    "new-customized-reward": ("new_reward", "CustomizedRewardScorer"),
}
```

**Step 3: apply reward.**

Apply your reward function by adding reward name in configuration:
```bash
python3 -m gerl.trainer.main_flowgrpo \
    data.reward_fn='["new-customized-reward"]' \
    ...
```

**Step4: tips and testing.**

You'd better have a unit test script for your scorer before applying your reward to run training, and add your testing in `tests/reward_score/run_reward_fns.sh`.

For example, `test_paddle_ocr_scorer` in `gerl/utils/reward_score/ocr.py` and `test_qwen_vl_ocr_vllm_scorer` in `gerl/utils/reward_score/vllm.py`.

## Examples
### Model-based or Rule-based Reward (CPU only)
`gerl/utils/reward_score/ocr.py`'s `PaddleOCRScorer` is an example for "PaddleOCR" model as OCR reward scorer.
1. In `__init__()` load model. Note that we must initialize and load model in `__init__()` once for speedup.
2. In `__call__()`, model infers recognized texts; and then compute reward values.

If you implement a rule-based reward, you only need to compute reward values in `__call__()`.

### API-serving Model-based Reward (e.g., vLLM, SGLang)
`gerl/utils/reward_score/vllm.py`'s `QwenVLOCRVLLMScorer` is an example for calling "Qwen2.5-VL" model via [vllm](https://github.com/vllm-project/vllm) as the OCR reward scorer.
1. In `__init__()` initialize vllm client setup.
2. In `__call__()`, prepare input prompts and call the model to get responses via vllm serving; then we compute reward values.
