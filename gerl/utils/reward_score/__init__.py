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


class DefaultComputeScore:
    """Compute the score for a given solution based on the reward_fn or data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        dict/float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.
    """

    def __init__(
        self,
        sandbox_fusion_url=None,
        concurrent_semaphore=None,
        memory_limit_mb=None,
    ) -> None:
        super().__init__()
        self.scorer = None
        self.sandbox_fusion_url = sandbox_fusion_url
        self.concurrent_semaphore = concurrent_semaphore
        self.memory_limit_mb = memory_limit_mb

    def get_scorer(
        self,
        data_source,
        extra_info=None,
    ) -> None:
        """Initialize once the scorer
        Args:
            data_source (str): The source dataset identifier which determines the scoring method.
            extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
        """
        reward_fn = extra_info.get("reward_fn", []) if extra_info else []
        if len(reward_fn) > 0:
            scorers_weight = [1 / len(reward_fn)] * len(
                reward_fn
            )  # TODO: TBD support custom weights, now use equal weights
            scorers = dict(zip(reward_fn, scorers_weight))
            from . import multi

            self.scorer = multi.MultiScorer(scorers)  # type: ignore
        else:
            print(
                "reward_fn is not specified, use default reward function for each data_source."
            )
            if data_source in [
                "ocr",
            ]:
                from . import ocr

                # init OCR model scorer
                self.scorer = ocr.PaddleOcrScorer()  # type: ignore
            else:
                print(
                    f"Unrecognized {data_source=}, use `jpeg-imcompressibility` as default."
                )
                from . import jpeg_imcompressibility

                self.scorer = jpeg_imcompressibility.JpegImcompressibilityScorer()  # type: ignore

    def __call__(
        self,
        data_source,
        solution_str,
        ground_truth,
        extra_info=None,
        **kwargs,
    ):
        if self.scorer is None:
            self.get_scorer(data_source, extra_info=extra_info)
        res = self.scorer(solution_str, ground_truth)

        if isinstance(res, dict):
            return res
        elif isinstance(res, int | float | bool):
            return float(res)
        elif isinstance(res, list):
            if len(res) == 1:
                return float(res[0])
            else:
                return [float(r) for r in res]


__all__ = ["DefaultComputeScore"]
