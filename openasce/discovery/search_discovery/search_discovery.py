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

from typing import Callable, Tuple, Union

import numpy as np

from openasce.discovery.causal_graph import CausalGraph
from openasce.discovery.discovery import Discovery
from openasce.discovery.search_discovery.search_strategy import Strategy
from openasce.utils.logger import logger


class CausalSearchDiscovery(Discovery):
    """Execute the causal inference by search method

    Attributes:

    """

    def __init__(self) -> None:
        """Constructor

        Arguments:

        Returns:

        """
        super().__init__()

    def fit(self, *, X: Union[np.ndarray, Callable], **kwargs) -> None:
        """Feed the sample data

        Arguments:
            X (num of samples, features or callable returning np.ndarray): samples
        Returns:

        """
        self._data = X() if callable(X) else X
        if isinstance(self._data, np.ndarray):
            if self.node_names and len(self.node_names) == self._data.shape[1]:
                pass
            elif self.node_names:
                raise ValueError(
                    f"The number of node does NOT match the column num of samples."
                )
            else:
                logger.info(
                    f"No node name specified. Use arbitrary names like x0, x1..."
                )
                self.node_names = [f"x{i}" for i in range(self._data.shape[1])]
        elif isinstance(self._data, dict):
            self.node_names = self._data.get("node_names")
            self._data = self._data.get("data")
        elif isinstance(self._data, tuple):
            self.node_names = [self._data[0]]
            self._data = self._data[1]
        else:
            raise ValueError(f"No reasonal input data. {type(self._data)}")
        strategy = Strategy(node_names=self.node_names, **kwargs)
        self._graph, self._graph_score = strategy.run(data=self._data)

    def get_result(self) -> Tuple[CausalGraph, float]:
        """Get the causal graph sample data

        Arguments:
            X (num of samples, features or callable returning np.ndarray): samples
        Returns:

        """
        return (self._graph, self._graph_score)
