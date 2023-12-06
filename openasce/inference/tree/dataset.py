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

import numpy as np
import pandas as pd
from pyhocon import ConfigFactory

from .reflect_utils import get_class
from .utils import to_row_major, logger


class Dataset(object):
    """Abstract interface of class dataset"""

    def __init__(self):
        pass

    def __len__(self):
        return self.features.shape[0]

    @staticmethod
    def new_instance(conf):
        data_conf = conf.get('dataset', conf)
        cls_name = data_conf.get('type', 'dataset.CSVDataset')
        return get_class(cls_name).new_instance(conf)

    def read(self, filename):
        pass

    def sub_dataset(self, index=None):
        """
        Abstract interface of sub-sampling

        Arguments:
            index (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def description(self, detail: bool = False) -> None:
        """
        description the dataset

        Arguments:
            detail (bool, optional): [description]. Defaults to False.
        """
        n_ins, n_feat = self.features.shape
        n_y_len = self.targets.shape[1]
        # calculate treatment distinct count
        treats = np.unique(self.treatment)
        logger.info(f'#inst: {n_ins}')
        logger.info(f'#feat: {n_feat}')
        logger.info(f'#time serise length: {n_y_len}')
        logger.info(f'#treatments : {len(treats)}')

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def features(self):
        raise NotImplementedError

    @property
    def treatment(self):
        raise NotImplementedError

    @property
    def feature_columns(self):
        if hasattr(self, 'used_features'):
            return getattr(self, 'used_features')
        elif isinstance(self.features, pd.DataFrame):
            return self.features.columns
        else:
            raise RuntimeError('There is no attribute `feature_columns`!')


class PsudoDataset(Dataset):
    """
    A Psudo Dataset to wrap for the numpy formatting data.

    Arguments:
        features (np.ndarray, optional): features. Defaults to None.
        outcome (np.ndarray, optional): outcome. Defaults to None.
        treatment (np.ndarray, optional): treatment. Defaults to None.
        conf (_type_, optional): configure. Defaults to None.
    """
        
    def __init__(self, features: np.ndarray=None, outcome: np.ndarray=None, treatment: np.ndarray=None, conf=None):

        self._features = to_row_major(features.copy('C')) if features is not None else None
        self._treatment = to_row_major(treatment.copy('C')) if treatment is not None else None
        self._outcome = to_row_major(outcome.copy('C')) if outcome is not None else None
        if conf is None:
            feature_columns = [f'X{i}' for i in range(features.shape[1])]
            conf = ConfigFactory.from_dict({'dataset':{'feature': feature_columns}})
        self.conf = conf
        self.used_features = self.conf.dataset.feature

    @property
    def features(self):
        return self._features
    
    @property
    def treatment(self):
        return self._treatment
    
    @property
    def targets(self):
        return self._outcome

    @property
    def weight(self):
        return np.ones_like(self.treatment)

    @property
    def feature_columns(self):
        return self.used_features

    def sub_dataset(self, index=None, cols=None) -> Dataset:
        """
        Create a sub-dataset.

        Arguments:
            index: Indices of the samples to include in the sub-dataset.
            cols: Columns to include in the sub-dataset.

        Returns:
            The sub-dataset.

        """
        if index.dtype in (pd.BooleanDtype, np.bool):
            assert index.shape[0] == self.n_inst
            index = np.where(index)[0]
        data_conf = self.conf

        if cols is None:
            idx = index
            feature_columns = self.feature_columns
        else:
            cols = np.asarray(cols, dtype=int)
            idx = np.ix_(index, cols)
            feature_columns = self.feature_columns[cols]

        _feature = self._features[idx].copy()
        _treatment = self._treatment[idx].copy()
        _outcome = self._outcome[idx].copy()
        data = PsudoDataset(_feature, _outcome, _treatment, conf=data_conf)
        data.used_features = feature_columns
        if hasattr(self, 'bin_features') and self.bin_features is not None:
            # bin features : DataFrame
            if cols is None:
                cols = self.bin_features.columns
            data.bin_features = self.bin_features.loc[index, cols]
        return data
