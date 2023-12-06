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
from functools import reduce
from typing import Dict, Iterable, List

import numpy as np

from openasce.discovery.causal_graph import CausalGraph
from openasce.discovery.discovery import Discovery
from openasce.discovery.graph_node_form import GraphNodeForm
from openasce.inference.inference_model import InferenceModel
from openasce.utils.logger import logger


class GraphInferModel(InferenceModel):
    """The inference using the causal graph

    Attributes:
        graph: The causal graph. If not set, the class will try to find it out if discovery is available.
        column_names: all names of sample
        treatment_name: treatment column name in column_names
        label_name: target column name in column_names
    """

    def __init__(
        self,
        *,
        graph: CausalGraph = None,
        column_names: List[str] = None,
        treatment_name: str = None,
        label_name: str = None,
        num_iteration=20,
    ) -> None:
        """
        Arguments:
            graph: causal graph
            column_names: all names of column
            treatment_name: the name of treatment column
            label_name: the name of target name
        """
        super().__init__()
        self._graph = graph
        self._column_names = column_names
        self._treatment_name = treatment_name
        self._label_name = label_name
        self._discovery = None
        self._data = None
        self._num_iteration = num_iteration
        self._label_value = None

    @property
    def data(self):
        assert self._data is not None, f"Must have sample data."
        return self._data

    @property
    def graph(self):
        assert self._graph is not None, "The graph object should be set"
        return self._graph

    @graph.setter
    def graph(self, value):
        assert self._graph is None, "The graph object should be set once only"
        self._graph = value
        # graph is available, set the column names using graph columns
        self.column_names = list(self.graph.names_to_index.keys())

    @property
    def column_names(self):
        """All nodes' name.
        Note: should include the treatment node and label node.
        """
        assert self._column_names is not None, "The column names should be set"
        return self._column_names

    @column_names.setter
    def column_names(self, value: List[str]):
        assert self._column_names is None, "The column names should be set once only"
        self._column_names = value

    @property
    def treatment_name(self):
        assert self._treatment_name is not None, "The treatment name should be set"
        return self._treatment_name

    @treatment_name.setter
    def treatment_name(self, value: str):
        assert (
            self._treatment_name is None
        ), "The treatment name should be set once only"
        self._treatment_name = value

    @property
    def label_name(self):
        assert self._label_name is not None, "The label name should be set"
        return self._label_name

    @label_name.setter
    def label_name(self, value: str):
        assert self._label_name is None, "The label name should be set once only"
        self._label_name = value

    @property
    def discovery(self) -> Discovery:
        assert self._discovery is not None, "The discovery object should be set"
        return self._discovery

    @discovery.setter
    def discovery(self, value: Discovery):
        self._discovery = value

    def fit(
        self,
        *,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray] = None,
        T: Iterable[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Feed the sample data to train the graph

        Arguments:
            X: All features of the samples including the treatment and the label node.
            Y: Ignore in causal graph inference
            T: Ignore in causal graph inference.

        Returns:

        """
        if Y is not None or T is not None:
            logger.info(
                f"All columns used in the casual graph discovery should be in X and Y/T is ignore"
            )
        self._data = np.vstack(list(iter(X)))
        if self.graph:
            if not self.graph.para:
                self.graph.calculate_parameter(data=self._data)
        elif self._discovery:
            self.column_names = self.discovery.node_names
            logger.info(f"Begin to discover the causal graph.")
            self.discovery.fit(self._data)
            self.graph, _ = self.discovery.get_result()
            if not self.graph.para:
                self.graph.calculate_parameter(data=self._data)
        else:
            raise ValueError(f"There is neither causal graph nor discovery.")

    def estimate(
        self,
        *,
        X: Iterable[np.ndarray] = None,
        Y: Iterable[np.ndarray] = None,
        T: Iterable[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Feed the sample data and estimate the outcome on the samples

        Arguments:
            X: Features of the samples.
            Y: Ignore in causal graph inference
            T: Ignore in causal graph inference
            kwargs: {'treat_value': treat_value, 'label_value':label_value}
        Returns:

        """
        if (
            not self.column_names
            or not self.treatment_name
            or not self.label_name
            or self.treatment_name not in self.column_names
            or self.label_name not in self.column_names
        ):
            raise ValueError(
                f"Either label or treatment is not set, or treat or label is not in columns."
            )
        self._data = np.vstack(list(iter(X))) if X else self._data

        self._label_value = kwargs.get(InferenceModel.LABEL_VALUE, None)
        treatment_value = kwargs.get(
            InferenceModel.TREATMENT_VALUE
        )  # Support one treatment value for now
        condition = dict(
            map(
                lambda p: (p[0], p[1] if isinstance(p[1], List) else [p[1]]),
                kwargs.get(InferenceModel.CONDITION_DICT_NAME, {}).items()
                if isinstance(kwargs.get(InferenceModel.CONDITION_DICT_NAME, {}), dict)
                else {},
            )
        )
        result = self._do_lbp(
            do_condition={
                self.treatment_name: treatment_value
                if isinstance(treatment_value, list)
                else [treatment_value]
            },
            condition=condition,
        )
        self._result = dict(
            map(lambda x: [self.graph.index_to_names.get(x[0]), x[1]], result.items())
        )

    def get_result(self):
        """Get the estimated result

        The sub-class should implement this routine and runtime invokes it.

        Arguments:

        Returns:
            The estimation result.
        """
        if self._label_value:
            self._result.get(self.label_name)
        else:
            return self._result

    def output(self):
        """Output the estimated result to files

        The sub-class should implement this routine and runtime invokes it.

        Arguments:

        Returns:

        """
        raise NotImplementedError(f"Not implement for abstract class")

    def _do_lbp(self, *, do_condition: Dict, condition: Dict):
        """compute the under the treatment

        Argument:
            do_condition: treatment node and its value
            condition: node and its value need to compute the outcome
        """
        n_iterr = self._num_iteration
        data = self._data
        graph = self.graph

        s_ij, s_ji, p_node, p_factor, node_link = {}, {}, {}, {}, {}
        do_node_list = list(do_condition.keys())
        assert len(do_node_list) == 1, "Only one DO node is supported for now"
        do_node_n = [graph.names_to_index[l] for l in do_node_list]
        all_conditions = copy.deepcopy(condition)
        all_conditions.update(do_condition)
        node_link = dict(
            map(
                lambda i: [
                    i,
                    set(
                        filter(
                            lambda j: j == i
                            or (j not in do_node_n and i in graph.parents[j]),
                            range(graph.n),
                        )
                    ),
                ],
                range(graph.n),
            )
        )
        for i in range(graph.n):
            values = np.unique(data[:, i])
            tmp_node = [[v_, 1 / len(values)] for v_ in values]
            p_node[i] = GraphNodeForm(
                tmp_node,
                columns=[graph.index_to_names[i], GraphNodeForm.SCORE_COLUMN_NAME],
            )
            p_node[i] = self._strict_to_condition(all_conditions, p_node[i])
            decrete_ = [
                [v_, list(data[:, i]).count(v_) / len(data[:, i])] for v_ in values
            ]
            s_ij[i] = {}
            s_ji[i] = {}
            par_set = set([i]) if i in do_node_n else set(list(graph.parents[i]) + [i])
            for k in par_set:
                values_ = np.unique(data[:, k])
                tmp_decrete_ = [
                    [v_, list(data[:, k]).count(v_) / len(data[:, k])] for v_ in values_
                ]
                s_ij[i][k] = GraphNodeForm(
                    tmp_decrete_,
                    columns=[graph.index_to_names[k], GraphNodeForm.SCORE_COLUMN_NAME],
                )
                s_ij[i][k] = self._strict_to_condition(all_conditions, s_ij[i][k])
            for k in node_link[i]:
                s_ji[i][k] = GraphNodeForm(
                    decrete_,
                    columns=[graph.index_to_names[i], GraphNodeForm.SCORE_COLUMN_NAME],
                )
                s_ji[i][k] = self._strict_to_condition(all_conditions, s_ji[i][k])
        p_factor = copy.deepcopy(graph.para)
        for node_i in p_factor.keys():
            p_factor[node_i] = self._strict_to_condition(
                all_conditions, p_factor[node_i]
            )
            if node_i in do_node_n:  # it is do_node
                p_factor[node_i].set_groupby_sum(do_node_list[0])

        run_count = 0
        ori_dis = copy.deepcopy(p_node)
        while run_count < n_iterr:
            for num in range(graph.n):
                for out_j in s_ji[num].keys():
                    s_ji[num][out_j] = self._update_normalize(
                        reduce(
                            lambda x, y: self._update_multipy(x, y),
                            map(
                                lambda in_j: s_ij[out_j][in_j],
                                filter(lambda in_j: in_j != num, s_ij[out_j].keys()),
                            ),
                            p_factor[out_j],
                        )
                    )
            for num in range(graph.n):
                for out_i in s_ij[num].keys():
                    s_ij[num][out_i] = self._update_normalize(
                        reduce(
                            lambda x, y: self._update_multipy(x, y),
                            map(
                                lambda in_i: s_ji[out_i][in_i],
                                filter(lambda in_i: in_i != num, s_ji[out_i].keys()),
                            ),
                            p_node[out_i],
                        )
                    )
            error_list = []
            for num in range(graph.n):
                tmp = self._update_normalize(
                    reduce(
                        lambda x, y: self._update_multipy(x, y),
                        map(lambda out_j: s_ji[num][out_j], s_ji[num].keys()),
                        p_node[num],
                    )
                )
                error = ori_dis[num].get_score_deviation(tmp)
                error_list.append(error)
                ori_dis[num] = tmp
            if sum(error_list) < 0.00001:
                break
            run_count += 1
        logger.info(f"Finish the lbp process. ")
        return ori_dis

    def _strict_to_condition(self, condition, p_r):
        column_list: List[str] = p_r.columns
        for key, value in condition.items():
            if key in column_list:
                p_r.set_flag_zero(key, value)
        p_r.set_norm()
        return p_r

    def _update_multipy(self, form_a: GraphNodeForm, form_b: GraphNodeForm):
        def _update_multipy_internal(form1: GraphNodeForm, form2: GraphNodeForm, flag):
            # Pick up the key except GraphNodeForm.SCORE_COLUMN_NAME
            unique_key_index = 1 if form2.score_column_index == 0 else 0
            unique_key = form2.columns[unique_key_index]
            form1.multiply_score_column(unique_key, form2)
            form1.sort_by_column(unique_key)

            res = {}
            res_r = []
            for l_data in form1.data:
                k_st = "".join(
                    map(
                        lambda y: str(int(l_data[y[0]])),
                        filter(
                            lambda x: not (
                                (x[1] == unique_key and flag)
                                or (x[1] == GraphNodeForm.SCORE_COLUMN_NAME)
                            ),
                            enumerate(form1.columns),
                        ),
                    )
                )
                res[k_st] = res.get(k_st, [])
                res[k_st].append(l_data)
            unique_key_index = form1.index(unique_key)
            columns_list = form1.columns
            for d_dt in res.values():
                dc = d_dt[0]
                dc[form1.score_column_index] = sum(
                    map(lambda x: x[form1.score_column_index], d_dt)
                )
                if flag:
                    at_list = [True for i in range(dc.size)]
                    at_list[unique_key_index] = False
                    dc = dc[at_list]
                res_r.append(dc)
            if flag:
                columns_list.pop(unique_key_index)
            return GraphNodeForm(res_r, columns_list)

        flag = not (set(list(form_a.columns)) == set(list(form_b.columns)))
        res_r = _update_multipy_internal(copy.deepcopy(form_a), form_b, flag)
        return res_r

    def _update_normalize(self, node: GraphNodeForm):
        node.set_norm()
        return node
