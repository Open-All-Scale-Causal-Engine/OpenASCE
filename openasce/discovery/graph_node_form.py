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
from typing import List

import numpy as np

from openasce.utils.logger import logger


class GraphNodeForm(object):
    SCORE_COLUMN_NAME = "node_score_value"

    def __init__(self, input_data: List[List[float]], columns: List[str]) -> None:
        self._columns = copy.deepcopy(columns)  # ['col1', 'col2']
        if GraphNodeForm.SCORE_COLUMN_NAME in columns:
            self._data = np.array(input_data, dtype=np.float64)  # np.ndarray
        else:
            self._columns.append(GraphNodeForm.SCORE_COLUMN_NAME)
            self._data = np.array(input_data, dtype=np.float64)  # np.ndarray
            self._data = np.column_stack((self._data, np.zeros(self._data.shape[0])))
        self._score_column_index = self._columns.index(GraphNodeForm.SCORE_COLUMN_NAME)

    @property
    def size(self):
        return len(self._data)

    @property
    def columns(self):
        return self._columns

    @property
    def data(self):
        return self._data

    @property
    def score_column_index(self):
        return self._score_column_index

    def index(self, key: str):
        return self._columns.index(key)

    def set_flag_zero(self, key: str, value_list: List[int]) -> None:
        """set score column to 0 if the value of key column is not in input value_list

        Arguments:
            key: the column name
            value_list: the values need to be set
        Returns:
            None
        """
        key_index = self._columns.index(key)
        score_column_index = self._score_column_index
        curr_data = self._data
        for i, row in enumerate(curr_data):
            if int(row[key_index]) not in value_list:
                curr_data[i, score_column_index] = 0

    def set_norm(self) -> None:
        """normalize the value of score column"""
        score_column_index = self._score_column_index
        curr_data = self._data
        prob_sum = (
            curr_data[:, score_column_index].sum() + 0.00000001
        )  # avoid zero as divisor
        for row in curr_data:
            row[score_column_index] /= prob_sum

    def multiply_score_column(self, key: str, ext) -> None:
        """multiply ext's score column to local score column for same key column's value

        Arguments:
            key: the column name
            ext (GraphNodeForm): another GraphNodeForm
        Returns:
            None
        """
        key_index = self._columns.index(key)
        curr_data = self._data
        score_column_index = self._score_column_index
        external_key_index = ext._columns.index(key)
        external_data = ext._data
        ext_score_column_index = ext._score_column_index
        for row in curr_data:
            for ext_row in external_data:
                if row[key_index] == ext_row[external_key_index]:
                    row[score_column_index] *= ext_row[ext_score_column_index]

    def sort_by_column(self, key: str) -> None:
        """sort specified column

        Arguments:
            key: the column name

        Returns:
            None
        """
        key_index = self._columns.index(key)
        curr_data = self._data
        self._data = np.array(sorted(curr_data, key=lambda x: x[key_index]))

    def get_score_deviation(self, addition):
        """multiply ext's score column to local score column for same key column's value

        Arguments:
            addition: Another GraphNodeForm used to calculate the deviation
        Returns:
            Calculation result
        """
        curr_data = self._data
        score_column_index = self._score_column_index
        external_data = addition.data
        ext_score_column_index = addition._score_column_index
        t = np.abs(
            curr_data[:, score_column_index : score_column_index + 1]
            - external_data[:, ext_score_column_index : ext_score_column_index + 1]
        )
        return t.sum()

    def get_score_value(self, target_key: str, target_value: int):
        """multiply ext's score column to local score column for same key column's value

        Arguments:
            target_key: the column name
            target_value: the column value

        Returns:

        """
        key_index = self._columns.index(target_key)
        curr_data = self._data
        score_column_index = self._score_column_index
        for row in curr_data:
            if int(row[key_index]) == target_value:
                return row[score_column_index]
        raise ValueError(f"Not target value exists")

    def set_groupby_sum(self, key: str):
        """multiply ext's score column to local score column for same key column's value

        Arguments:
            key: the column name

        Returns:

        """
        key_index = self._columns.index(key)
        curr_data = self._data
        score_column_index = self._score_column_index
        ac = {}
        for row in curr_data:
            if int(row[key_index]) in ac:
                ac[int(row[key_index])] += row[score_column_index]
            else:
                ac[int(row[key_index])] = row[score_column_index]
        result_data = np.zeros(shape=(len(ac), 2), dtype=np.float64)
        line_num = 0
        for k1, value in ac.items():
            result_data[line_num] = np.array([k1, value], dtype=np.float64)
            line_num += 1
        self._data = result_data
        self._columns = [key, GraphNodeForm.SCORE_COLUMN_NAME]
        self._score_column_index = self._columns.index(GraphNodeForm.SCORE_COLUMN_NAME)

    def __str__(self):
        np.set_printoptions(threshold=5000, suppress=True)
        return self.columns.__str__() + "\n" + self._data.__str__() + "\n"
