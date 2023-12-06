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

from pyhocon import ConfigTree
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from .utils import value_bin_parallel, find_bin_parallel, TRACE, to_row_major
from .information import CausalDataInfo


class BinMapper(KBinsDiscretizer):
    """A class for binning numerical features."""
    def __init__(self, conf: ConfigTree):
        self.info = CausalDataInfo(conf)
        self._binmaper_cpp: list = None

    def transform(self, X):
        """
        Transform the input features using the bin mapper.

        Arguments:
            X: Input features.

        Returns:
            The transformed features.

        """
        return value_bin_parallel(X, self._binmaper_cpp)

    def fit(self, X, y=None):
        """
        Fit the bin mapper on the input features.

        Arguments:
            X: Input features.
            y: The target variable (not used).

        Returns:
            The fitted bin mapper object.
        """
        xshape = X.shape
        assert len(xshape) == 2, f'`X` must be 2-dimension!'
        self._binmaper_cpp = find_bin_parallel(X, self.info.n_bins, self.info.min_point_per_bin,
                                             self.info.min_point_per_bin, True)
        self.description()
        return self

    def description(self):
        """Print the description of the bin mapper."""
        TRACE(f'{"*"*43}description bin{"*"*43}')
        TRACE(f'*{len(self._binmaper_cpp)} features*')
        TRACE(f'*number of bins:{[len(b.GetUpperBoundValue()) for b in self._binmaper_cpp]}')
        TRACE(f'{"*"*100}')

    def inverse_transform(self, Xt, index: int = None):
        """
        Inverse transform the transformed features to the original values.

        Arguments:
            Xt: Transformed features.
            index: Index of the feature to inverse transform.

        Returns:
            The inverse transformed features.

        """
        if index is not None:
            assert len(self._binmaper_cpp) > index and index >= 0, f'index must between [0, {len(self._binmaper_cpp)})!'
            return self._binmaper_cpp[index].BinToValue(Xt)
        raise NotImplementedError

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the bin mapper on the input features and transform them.

        Arguments:
            X: Input features.
            y: The target variable (not used).
            fit_params: Additional parameters for fitting.

        Returns:
            The transformed features.

        """
        self.fit(X)
        return self.transform(X)
    
    def fit_dataset(self, data):
        """
        Fit the bin mapper on the dataset.

        Arguments:
            data: Dataset object containing the input features.

        """
        x = to_row_major(data.features)
        if self.is_fit is False:
            self.fit(x)
        bin_features = self.transform(x)
        bin_features = pd.DataFrame(bin_features, columns=data.feature_columns)
        data.bin_features = bin_features

    @property
    def is_fit(self):
        """
        Check if the bin mapper is fit.

        Returns:
            True if the bin mapper is fit, False otherwise.

        """
        return self._binmaper_cpp is not None

    @property
    def upper_bounds(self):
        """
        Get the upper bounds of the bins.

        Returns:
            The upper bounds of the bins.

        """
        return np.asfarray([m.GetUpperBoundValue() for m in self._binmaper_cpp])
