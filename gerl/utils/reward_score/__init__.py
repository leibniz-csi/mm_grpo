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


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    reward_fn = extra_info.get("reward_fn", []) if extra_info else []
    if len(reward_fn) > 0:
        scorers_weight = [1 / len(reward_fn)] * len(
            reward_fn
        )  # TODO: TBD support custom weights
        scorers = dict(zip(reward_fn, scorers_weight))
        from . import multi

        res = multi.compute_score(solution_str, ground_truth, scorers)
    else:
        print(
            "reward_fn is not specified, use default reward function for each data_source."
        )
        if data_source in [
            "ocr",
        ]:
            from . import ocr

            res = ocr.compute_score(solution_str, ground_truth)

        else:
            print(
                f"Unrecognized {data_source=}, use jpeg-imcompressibility as default."
            )
            from . import jpeg_imcompressibility

            res = jpeg_imcompressibility.compute_score(solution_str, ground_truth)

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    elif isinstance(res, list):
        if len(res) == 1:
            return float(res[0])
        else:
            return [float(r) for r in res]


__all__ = ["default_compute_score"]
