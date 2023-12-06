#    Copyright 2023 AntGroup CO., Ltd.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from typing import Callable, List, Union

import numpy as np

from openasce.core.runtime import Runtime


class Discovery(Runtime):
    """Discovery Class

    Base class of the causal discovery

    Attributes:
        node_names (List[str]): the name of graph node, which should be set before fit

    """

    def __init__(self) -> None:
        super().__init__()
        self._node_names = []

    def fit(self, *, X: Union[np.ndarray, Callable], **kwargs) -> None:
        """Feed the sample data and search the causal relation on them

        Arguments:
            X: Features of the samples.

        Returns:
            None
        """
        raise NotImplementedError(f"Not implement for abstract class")

    def get_result(self):
        """Output the causal graph

        Returns:
            None
        """
        raise NotImplementedError(f"Not implement for abstract class")

    @property
    def node_names(self):
        return self._node_names

    @node_names.setter
    def node_names(self, value: List[str]):
        self._node_names = value
