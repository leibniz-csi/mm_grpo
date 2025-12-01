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
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

from dataclasses import dataclass, field

from tensordict import TensorDict
from verl import DataProto as verlDataProto
from verl.protocol import union_numpy_dict, union_tensor_dict
from verl.utils.py_functional import union_two_dict

__all__ = ["DataProto"]


@dataclass
class DataProto(verlDataProto):
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if

        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        if self.batch is not None and other.batch is not None:
            self.batch = union_tensor_dict(self.batch, other.batch)
        elif (self.batch or other.batch) is not None:
            self.batch = self.batch or other.batch
        self.non_tensor_batch = union_numpy_dict(
            self.non_tensor_batch, other.non_tensor_batch
        )
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self
