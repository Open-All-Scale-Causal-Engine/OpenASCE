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

import copy
import itertools
from collections import Counter
from typing import Dict, List, Union

import numpy as np
from scipy.special import gammaln

from openasce.discovery.graph_node_form import GraphNodeForm
from openasce.utils.logger import logger


class CausalGraph(object):
    """Causal Graph Class

    Represent the casual graph

    """

    DEFAULT_COLUMN_NAME_PREFIX = "x"

    def __init__(self, names=[], bn=None, w: np.ndarray = None):
        """Constructor

        Arguments:
            names: the node names
            bn: basic causal graph
            w: the connection matrix for causal graph

        """
        self.para = None
        self.parents = {}  # {c1:[p1, p2],c2:[p2,p3]....}
        self.names_to_index = {}
        self.index_to_names = {}
        self.n = 0
        self.index_exclude = []
        if bn is not None:
            self.copy(bn)
        else:
            if names:
                self.names_init(names)
            if w is not None:
                if self.names_to_index and self.index_to_names and self.parents:
                    pass
                else:
                    self.names_init(
                        [
                            self.DEFAULT_COLUMN_NAME_PREFIX + str(i)
                            for i in range(w.shape[0])
                        ]
                    )
                nz = w.nonzero()
                for _ in map(lambda x: self.add_edge(x[0], x[1]), zip(nz[0], nz[1])):
                    pass

    def names_init(self, names: List[str]) -> None:
        """Initialize the graph with feature names

        initialize the names_to_index and index_to_names attributes
        initialize parents[i] = set() (no edges for the moment)

        Arguments:
            names (list of string): the names of the nodes

        Returns:
            None
        """
        tmp_names = copy.deepcopy(names)
        self.names_to_index = {name: index for index, name in enumerate(names)}
        self.index_to_names = {index: name for index, name in enumerate(tmp_names)}
        self.n = len(self.names_to_index)
        for i in range(self.n):
            self.parents[i] = set()

    def parents_exclude(self, name_list: List[str]) -> None:
        for name in name_list:
            self.index_exclude.append(self.names_to_index[name])

    def random_init(self, max_parents: int = None) -> None:
        """Add edges randomly

        For each node, pick a random number of the desired number of parents.
        Then, for each candidate, pick another random number. In average,
        the node will have the desired number of parents.

        Arguments:
            max_parents: maximal number of one node's parents
        """
        max_parents = max_parents if max_parents else self.n - 1

        for i in range(self.n):
            nparents = np.random.randint(0, max_parents + 1)
            p = nparents / (self.n + 1.0)
            for j in range(self.n):
                if j != i and np.random.uniform() < p:
                    self.add_edge(j, i)

    def merge(
        self, g1, g2, p1=1, p2=1, max_parents: int = None, mut_rate: float = 0.0
    ) -> None:
        """Pick up edges from both g1 and g2 according to some random policy

        Arguments:
            g1 (CausalGraph)
            g1 (CausalGraph)
            p1 (float in [0,1]): proba of an edge in g1 being in self
            p2 (float in [0,1]): proba of an edge in g2 being in self
                p1 + p2 = 1
            max_parents (int)

        """
        # merge randomly the two graphs
        self.random_merge(g1, g2, p1, p2)

        # introduce mutations
        self.mutate(mut_rate)

        # remove extra parents
        self.remove_extra_parents(max_parents)

    def random_merge(self, g1, g2, p1, p2) -> None:
        """Creates graph from edges both in g1 and g2. Adds edges according to proba p1 and p2

        Arguments:
            g1 (CausalGraph)
            g1 (CausalGraph)
            p1 (float in [0,1]): proba of an edge in g1 being in self
            p2 (float in [0,1]): proba of an edge in g2 being in self
        """
        for i, js in g1.parents.items():
            for j in js:
                if np.random.uniform() < p1 or p1 == 1:
                    self.add_edge(j, i)
        for i, js in g2.parents.items():
            for j in js:
                if np.random.uniform() < p2 or p2 == 1:
                    self.add_edge(j, i)

    def mutate(self, mut_rate: float = 0) -> None:
        """Introduces new edges with a probability mut_rate

        Arguments:
            mut_rate (float in [0,1]): proba of mutation
        """
        if mut_rate != 0:
            """Do mutation like the following code snippet
            for i in range(self.n):
                for j in range(self.n):
                    p = np.random.uniform()
                    if p < mut_rate:
                        if p < mut_rate / 2:
                            self.add_edge(i, j)
                        else:
                            self.remove_edge(i, j)
            """
            for _ in map(
                lambda x: self.add_edge(x[0], x[1])
                if x[2] < 0.25
                else self.remove_edge(x[0], x[1]),
                filter(
                    lambda x: x[2] <= 0.5,
                    map(
                        lambda x: x + (np.random.uniform(),),
                        itertools.product(self.n, self.n),
                    ),
                ),
            ):
                pass

    def remove_extra_parents(self, max_parents: int = None) -> None:
        """Removes extra edges if does not respect max parents constraint

        Arguments:
            max_parents: the maximal number of the node's parents
        """
        if max_parents is not None:
            for i, js in self.parents.items():
                if len(js) > max_parents:
                    indices = np.random.permutation(range(len(js)))
                    for j in indices[0 : len(js) - max_parents]:
                        self.remove_edge(j, i)

    def num_save(self, file_name: str) -> None:
        """
        Saves the graph in number format

        Example
            parent1, child1
            parent2, child2

        Arguments:
            file_name: saved file path
        """
        with open(file_name, "w") as f:
            for child_index, parents in self.parents.items():
                for parent_index in parents:
                    f.write(f"{parent_index},{child_index}\n")

    def save(self, file_path: str) -> None:
        """Saves the graph in the desired format

        Example
            parent1, child1
            parent2, child2
        Arguments:
            file_path: saved file path
        """
        with open(file_path, "w") as f:
            for child_index, parents in self.parents.items():
                for parent_index in parents:
                    parent = self.index_to_names.get(parent_index)
                    child = self.index_to_names.get(child_index)
                    f.write(f"{parent},{child}\n")

    def load(self, file_name: str) -> None:
        """Loads structure from file. See save method

        Arguments:
            file_name: the path of the file to be loaded
        """
        if not (self.names_to_index and self.index_to_names):
            name_set = set()
            # Go through the file to get all node names
            with open(file_name) as f:
                for line in f:
                    line = line.strip().split(",")
                    if len(line) == 2:
                        p = line[0].replace("'", "").replace('"', "").strip()
                        c = line[1].replace("'", "").replace('"', "").strip()
                        if p not in name_set:
                            name_set.add(p)
                        if c not in name_set:
                            name_set.add(c)
            self.names_to_index = {name: index for index, name in enumerate(name_set)}
            self.index_to_names = {index: name for index, name in enumerate(name_set)}
        with open(file_name) as f:
            for line in f:
                line = line.strip().split(",")
                if len(line) == 2:
                    p = line[0].replace("'", "").replace('"', "").strip()
                    c = line[1].replace("'", "").replace('"', "").strip()
                    logger.info(f"p={p}, c={c}")
                    p_index, c_index = self.names_to_index[p], self.names_to_index[c]
                    self.add_edge(p_index, c_index)

    def is_cyclic(self) -> bool:
        """Returns True if a cycle is found else False.

        Iterates over the nodes to find all the parents' parents, etc.
        A cycle is found if a node belongs to its own parent's set.

        """
        all_parents = copy.deepcopy(self.parents)
        update = True
        while update:
            update = False
            for i in range(self.n):
                parents = list(all_parents[i])
                nparents = len(parents)
                for p in parents:
                    all_parents[i].update(all_parents[p])
                if nparents != len(all_parents[i]):
                    update = True
                if i in all_parents[i]:
                    return True
        return False

    def copy(self, cg) -> None:
        """Copies the structure of cg inside self and erases everything else

        Arguments:
            cg (CausalGraph): model
        """
        self.index_to_names = copy.deepcopy(cg.index_to_names)
        self.names_to_index = copy.deepcopy(cg.names_to_index)
        self.n = cg.n
        self.parents = copy.deepcopy(cg.parents)

    def add_edge(
        self, parent: Union[int, str], child: Union[int, str], max_parents=None
    ) -> bool:
        """Adds edge if respects max parents constraint and does not create a cycle

        Arguments:
            parent (int): id of parent
            child (int): id of child
            max_parents (int): None means no constraints

        Returns
            True if actually added the edge and False means no way to add the edge
        """
        parent = self.names_to_index.get(parent) if isinstance(parent, str) else parent
        child = self.names_to_index.get(child) if isinstance(child, str) else child
        if (
            parent is None
            or child is None
            or parent >= self.n
            or child >= self.n
            or parent == child
        ):
            raise ValueError(f"Error parent or child")
        if max_parents is not None and len(self.parents[child]) >= max_parents:
            return False
        if child not in self.parents:
            self.parents[child] = set()
        self.parents[child].add(parent)
        if self.is_cyclic():
            logger.debug(
                f"The edge from {parent} to {child} produces a cycle and be refused"
            )
            self.remove_edge(parent, child)
            return False
        return True

    def remove_edge(self, parent: int, child: int, force: bool = True) -> None:
        try:
            self.parents[child].remove(parent)
        except Exception as e:
            if force:
                logger.debug(f"Exception happens in remove edge: \n{e}")
            else:
                raise e

    def score(self, data: np.ndarray, rd: Dict[int, int] = None) -> float:
        """Computes bayesian score of the structure given some data assuming uniform prior

        Example
            s = cg.score(data)

        Arguments:
            data: (nsamples, nfeatures)

        Returns
            s (float): bayesian score

        """
        s = 0
        r = rd if rd else self.compute_r(data)
        for i in range(self.n):
            s += self.score_node(i, data, r)
        return s

    def compute_r(self, data: np.ndarray) -> dict:
        """Compute the number of the value for each node

        Arguments:
            data (np array): (nsamples, nfeatures)
        Returns
            r (dict): r[i] = r_i
        """
        r = {}
        for i in range(self.n):
            r[i] = np.unique(data[:, i]).shape[0]
        return r

    def score_node(self, i, data: np.ndarray, r) -> float:
        """Compute the score of node i

        Arguments:
            i (int): node
            data (np array): (nsamples, nfeatures)
            r (dict of np array): r[i] = nb possible instances of i
        Returns
            s (float): contribution to log score of node i
        """
        m, m0 = Counter(), Counter()
        columns = [i] + list(self.parents.get(i))
        extracted_data = data[:, columns]
        # counting nb of each instance of (node, parents) and (parents)
        for sample in extracted_data:
            m[tuple(sample)] += 1
            m0[tuple(sample[1:])] += 1
        # Adding contribution to the score (assuming uniform prior)
        s: float = 0.0
        """Like following code snippet
        for c in m0.values():
            s -= gammaln(r[i] + c)
            s += gammaln(r[i])
        """
        stat_i = r[i]
        s -= sum(gammaln(stat_i + c) for c in m0.values())
        s += gammaln(stat_i) * len(m0.values())
        """Like following code snippet
        for c in m.values():
            s += gammaln(1 + c)
        """
        s += sum(gammaln(1 + c) for c in m.values())
        return s

    def calculate_parameter(self, data: np.ndarray, rd: Dict[int, int] = None):
        """Calculate the edge weight in the graph

        Arguments:
            data: samples
            rd: r[i] = r_i
        """
        r = rd if rd else self.compute_r(data)
        node_param = {}
        aux_para_cp = {}
        for i in self.parents.keys():
            if i not in node_param:
                node_param[i] = {}
            if i not in aux_para_cp:
                aux_para_cp[i] = {}
            list_par = [i] + list(self.parents[i])
            data_par = data[:, list_par]
            all_count = 0
            column_list = [self.index_to_names[k] for k in list_par]
            for data_line in data_par:
                tup_k = tuple(data_line)
                if tup_k in aux_para_cp[i].keys():
                    aux_para_cp[i][tup_k] += 1
                else:
                    aux_para_cp[i][tup_k] = 1
                name = ""
                for k in range(len(list_par)):
                    name += self.index_to_names[list_par[k]] + " = {} ".format(
                        data_line[k]
                    )
                if name in node_param[i].keys():
                    node_param[i][name] += 1
                else:
                    node_param[i][name] = 1
                all_count += 1
            count = 1
            for k_s in r.keys():
                if k_s in list_par:
                    count *= r[k_s]
            for tup_key in node_param[i].keys():
                node_param[i][tup_key] = (1 + node_param[i][tup_key]) / (
                    count + all_count
                )
            df_res = []
            for tup_key in aux_para_cp[i].keys():
                aux_para_cp[i][tup_key] = (1 + aux_para_cp[i][tup_key]) / (
                    count + all_count
                )
                list_tmp = list(tup_key)
                list_tmp.append(aux_para_cp[i][tup_key])
                df_res.append(list_tmp)
            column_list.append(GraphNodeForm.SCORE_COLUMN_NAME)
            p_ = GraphNodeForm(df_res, columns=column_list)
            node_param[i] = p_
        self.para = node_param
        return self.para
