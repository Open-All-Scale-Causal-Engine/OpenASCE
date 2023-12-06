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
import random
from typing import Iterable, List

import numpy as np

from openasce.core.runtime import Runtime
from openasce.inference.inference_model import InferenceModel
from openasce.utils.logger import logger


class Attribution(Runtime):
    """Attribution Class

    Attributes:

    """

    def __init__(
        self, *, threshold: float, max_step: int = 2, top_num: int = None
    ) -> None:
        """Constructor

        Argument:
            threshold: the score threshold
            max_step: the maximal step. For the attribution based on causal graph, that is the maximal node number.
            top_num: the accepted number of best options in each step, which is used in greedy attribution.
        """
        super().__init__()
        self._inferencer = None
        self._data = None
        self._threshold = threshold
        self._max_step = max_step
        self._top_num = top_num
        self._column_names = None
        self._treatment_name = None
        self._label_name = None
        self._label_value = None
        self._result = []

    @property
    def column_names(self):
        """All nodes' name.
        Note: should include the treatment node and label node.
        """
        assert self._column_names is not None, "column names should be set in advance"
        return self._column_names

    @column_names.setter
    def column_names(self, value: List[str]):
        assert self._column_names is None
        self._column_names = value

    @property
    def treatment_name(self):
        assert self._treatment_name is not None
        return self._treatment_name

    @treatment_name.setter
    def treatment_name(self, value: str):
        assert self._treatment_name is None
        self._treatment_name = value

    @property
    def label_name(self):
        assert self._label_name is not None
        return self._label_name

    @label_name.setter
    def label_name(self, value: str):
        assert self._label_name is None
        self._label_name = value

    @property
    def label_value(self):
        assert self._label_value is not None
        return self._label_value

    @label_value.setter
    def label_value(self, value):
        assert self._label_value is None
        self._label_value = value

    @property
    def inferencer(self) -> InferenceModel:
        """The inference object used to estimate the effect"""
        assert (
            self._inferencer is not None
        ), "Need to set the inferencer used to estimate the effect"
        return self._inferencer

    @inferencer.setter
    def inferencer(self, value: InferenceModel) -> None:
        self._inferencer = value
        if (
            hasattr(self._inferencer, "column_names")
            and self._inferencer.column_names
            and hasattr(self._inferencer, "treatment_name")
            and self._inferencer.treatment_name
            and hasattr(self._inferencer, "label_name")
            and self._inferencer.label_name
        ):
            logger.info(
                f"Setup the column name, treatment name and label name using inferencer"
            )
            self.column_names = self._inferencer.column_names
            self.treatment_name = self._inferencer.treatment_name
            self.label_name = self._inferencer.label_name

    def attribute(
        self,
        *,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray] = None,
        T: Iterable[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Feed the sample data to attribute.

        Arguments:
            X: Features of the samples.
            Y: Ignore for now and keep for future
            T: Ignore for now and keep for future
            kwargs: {'treat_value': treat_value}, maximization when treat_value

        Returns:
            None
        """
        if Y is not None or T is not None:
            logger.info(
                f"All columns used in the casual graph discovery should be in X and Y/T is ignore"
            )
        label_value, treatment_value = kwargs.get(
            InferenceModel.LABEL_VALUE
        ), kwargs.get(InferenceModel.TREATMENT_VALUE)
        self.inferencer.fit(X=X, Y=Y, T=T)
        data = self.inferencer.data
        logger.info(f"{self.column_names}\n{self.label_name}\n{self.treatment_name}")
        column_names = self.column_names
        exclusive_names = set(
            [
                self.label_name,
                self.treatment_name,
            ]
        )
        single_node_value_list = list(
            map(
                lambda y: {y[0]: y[1]},
                [
                    (column_names[col_index], v)
                    for col_index in range(len(column_names))
                    for v in np.unique(data[:, col_index])
                    if column_names[col_index] not in exclusive_names
                ],
            )
        )
        # Compute all nodes and values in first step
        conditions = copy.deepcopy(single_node_value_list)
        for step in range(self._max_step):
            logger.info(f"{conditions}")
            result_candidates = []
            for condition in conditions:
                logger.info(f"{condition}")
                self.inferencer.estimate(
                    condition=condition,
                    treatment_value=treatment_value,
                )
                res = self.inferencer.get_result().get(self.label_name)
                result_candidates.append(
                    (
                        condition,
                        res.get_score_value(
                            target_key=self.label_name, target_value=label_value
                        ),
                    )
                )
            self._result.extend(
                filter(lambda x: x[1] >= self._threshold, result_candidates)
            )
            logger.info(
                f"=========Step: {step}, total size of results:\n{len(self._result)}\n========="
            )
            result_candidates = sorted(
                list(filter(lambda x: x[1] < self._threshold, result_candidates)),
                key=lambda x: x[1],
                reverse=True,
            )
            result_candidates = (
                result_candidates[: self._top_num]
                if self._top_num
                else result_candidates
            )
            # Produce the new explored conditions
            random.shuffle(single_node_value_list)
            iser = iter(single_node_value_list)
            conditions = []
            for r in result_candidates:
                while True:
                    try:
                        node_value = next(iser)
                        if list(filter(lambda y: y not in r[0], node_value.keys())):
                            break
                    except StopIteration as e:
                        iser = iter(single_node_value_list)
                r[0].update(node_value)
                conditions.append(r[0])

    def get_result(self):
        """Get the result

        Returns:
            The attribution result.
        """
        return self._result
