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

"""
This script simulates async reward score computing via ray.remote,
validating the asynchrony of `compute_reward_async` in `gerl\\trainer\\ppo\\reward.py`.
"""

import time

import ray


def compute_reward(delay):
    """
    Sleep for `delay` seconds to simulate reward computation duration.
    """
    time.sleep(delay)

    return f"Return compute_reward with delay={delay}."


@ray.remote(num_cpus=1)
def compute_reward_async(delay):
    """
    Async version of compute_reward using ray.
    """
    finish_msg = compute_reward(delay)
    return finish_msg


def test_compute_reward_async():
    ray.init(include_dashboard=False)

    # Test asynchronous reward computation
    # Launch multiple async reward fns with varying delays
    start_time = time.time()
    futures = [compute_reward_async.remote(i + 1) for i in range(5)]
    end_time = time.time()
    async_launch_duration = end_time - start_time
    print(f"Launched async reward fns in {async_launch_duration:.2f} seconds.")
    # Get results
    start_time = time.time()
    for future in futures:
        msg = ray.get(future)
        print("Async:", msg)
    end_time = time.time()
    async_get_duration = end_time - start_time
    print(f"Fetched all async reward results in {async_get_duration:.2f} seconds.")
    print(
        f"Total async reward running in {(async_launch_duration + async_get_duration):.2f} seconds."
    )

    print("*" * 50)

    # Test synchronous reward computation
    # Run multiple sync reward fns with varying delays sequentially
    start_time = time.time()
    for i in range(5):
        print("Sync:", compute_reward(i + 1))
    end_time = time.time()
    sync_duration = end_time - start_time
    print(f"Run sync reward fns in {sync_duration:.2f} seconds.")  # around 15s

    assert async_launch_duration + async_get_duration < sync_duration, (
        "Async reward computation should be faster than sync."
    )


if __name__ == "__main__":
    test_compute_reward_async()
