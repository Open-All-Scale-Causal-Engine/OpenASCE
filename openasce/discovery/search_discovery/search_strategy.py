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

from typing import List, Tuple

import numpy as np

from openasce.discovery.causal_graph import CausalGraph
from openasce.utils.logger import logger


class Strategy(object):
    """General class to implement different structure learning methods

    Attributes
        edge_gain (float): the minimal gain of adding edge.
        target_name (str): the name of the node that will be label.
    """

    def __init__(self, node_names: List[str], **kwargs):
        """Contructor

        Arguments:
            node_names: the name of nodes
        """
        self.node_names = node_names
        self.strategy_name = "k2"
        self.edge_gain = kwargs.get("edge_gain", 20)
        self.target_name = kwargs.get("target_name", None)
        self.target_index = (
            self.node_names.index(self.target_name) if self.target_name else None
        )

    def run(self, data: np.ndarray, **kwargs) -> Tuple:
        """Run the actual strategy

        Arguments:
            data: the features of samples
            **kwargs (dict): dictionnary with method specific args

        Returns:

        """
        g, s = self.k2(data=data, **kwargs)
        logger.info(f"Best score is {s}")
        return g, s

    def best_parent(self, *, g, s, node_i, data, max_parents, r, s_i):
        """Search for best parent

        Returns g by adding to node i the best parent that maximizes the score

        Arguments:

        Returns:

        """
        found_new = False
        g_max = g
        s_max = s
        shuffle_no = np.random.permutation(range(g.n))
        if self.target_name:
            # The target node can't be any node's parent if set target, so remove it from the node candidate
            shuffle_no = np.delete(
                shuffle_no, np.where(shuffle_no == self.target_index)
            )
        shuffle_no_new = shuffle_no
        edge_gain = self.edge_gain
        for j in shuffle_no_new:
            if j != node_i and j not in g.parents[node_i]:
                g_work = CausalGraph(bn=g)
                if g_work.add_edge(j, node_i, max_parents):
                    # Try to add one edge between (j, node_i)
                    new_score = g_work.score_node(node_i, data, r)
                    logger.debug(f"new_score={new_score}")
                    s_new = s - s_i + new_score
                    if s_new > s_max + edge_gain:
                        found_new = True
                        g_max = g_work
                        s_max = s_new
        return g_max, s_max, found_new

    def k2(self, data: np.ndarray, **kwargs):
        """Implements k2 algorithm

        Agrument:
            data: the features of samples
        """
        names = self.node_names
        global_max_parents = (
            kwargs.get("max_parents")
            if kwargs.get("max_parents")
            else len(list(names)) / 2
        )
        max_parents = global_max_parents
        logger.info(
            f"current max parent number: {global_max_parents}, target_name={self.target_name}"
        )
        ordering = np.random.permutation(range(len(names)))
        if self.target_index:  # set target only so put target first one
            ordering = np.delete(ordering, np.where(ordering == self.target_index))
            ordering = np.insert(ordering, 0, self.target_index)
            max_parents = min(max_parents, len(list(names)) / 2)
        g = CausalGraph(names)
        global_s = g.score(data)
        logger.info(f"initial graph score:{global_s}")
        curr_data_r = g.compute_r(data)
        logger.info(f"graph curr_data_r={curr_data_r}")

        curr_pos = 0
        ordering_size = len(ordering)
        while curr_pos < ordering_size:
            node_i = ordering[curr_pos]
            s_i = g.score_node(node_i, data, curr_data_r)
            logger.info(f"Begin to explore node {node_i}, s_i={s_i}")
            curr_parent_count, found_new = 0, True
            while found_new and curr_parent_count < max_parents:
                g, global_s, found_new = self.best_parent(
                    g=g,
                    s=global_s,
                    node_i=node_i,
                    data=data,
                    max_parents=global_max_parents,
                    r=curr_data_r,
                    s_i=s_i,
                )
                curr_parent_count += 1
            max_parents = global_max_parents
            curr_pos += 1
        return g, global_s
