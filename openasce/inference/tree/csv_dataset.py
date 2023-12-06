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

import pandas as pd
import numpy as np

from .dataset import Dataset
from .utils import to_row_major


class CsvDataset(Dataset):
    """A Dataset interface for loading csv data"""

    def __init__(self, conf=None, **kwargs):
        super().__init__()

        self.conf = conf
        self.data_conf = self.conf.get('dataset', self.conf)
        self.bin_features = None

    def read(self, filename=None):
        if filename is None and self.conf is not None:
            filename = self.data_conf.get('train_data_path')[0]
        data = pd.read_csv(filename, index_col=0)
        feat_cols = self.data_conf.get_list('feature')
        treatment_info = self.data_conf.get('treatment_info', None)
        try:
            label_cols = [f.name for f in self.data_conf.get('label_columns')]
            treat_cols = [f.name for f in self.data_conf.get('treatment_columns')]
            weight_cols = [f.name for f in self.data_conf.get('weight_columns', [])]
        except:
            label_cols = [f for f in self.data_conf.get('label')]
            treat_cols = [self.data_conf.get('treatment')]
            weight_cols = [f for f in self.data_conf.get('weight', [])]
        for f in feat_cols + label_cols + treat_cols + weight_cols:
            if f not in data.columns:
                raise RuntimeError(f'feature `{f}` not exists in data!')
        # transform the treatment into [0, 1, 2, ....]
        if treatment_info is not None:
            treatment_map = {info[0]: np.int32(i) for i, info in enumerate(treatment_info)}
            data[treat_cols[0]] = data[treat_cols[0]].apply(lambda x: treatment_map[int(x)])
        self._data = data
        self.feat_cols = feat_cols
        self.label_cols = label_cols
        self.treat_cols = treat_cols
        self.weight_cols = weight_cols
        self.n_feat = len(feat_cols)

    def sub_dataset(self, index=None, cols=None, cols_y=[]) -> Dataset:
        if index.dtype in (pd.BooleanDtype, np.bool):
            assert index.shape[0] == self.n_inst
            index = np.where(index)[0]
        data_conf = self.conf
        data = CsvDataset(conf=data_conf)
        data.n_inst = index.shape[0]
        if cols is None:
            data.n_feat = self.n_feat
            cols = self.features.columns
        else:
            data.n_feat = len(cols)
        data.feat_cols = cols
        data.label_cols = self.label_cols
        data.treat_cols = self.treat_cols
        data.weight_cols = self.weight_cols
        data._data = self._data.iloc[index]
        return data

    @property
    def targets(self):
        return self._data[self.label_cols]

    @property
    def features(self):
        return self._data[self.feat_cols]
    
    @property
    def treatment(self):
        return self._data[self.treat_cols[0]]

    @property
    def weight(self):
        if len(self.weight_cols) > 0:
            return to_row_major(self._data[self.weight_cols[0]])
        return None

    @staticmethod
    def new_instance(conf):
        data_conf = conf.get('dataset', conf)
        data = CsvDataset(conf=conf)
        data.read(data_conf.get('data.path'))
        data.description()
        return data
