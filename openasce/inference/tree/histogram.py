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
from pyhocon import ConfigTree

from .utils import update_histogram, update_histograms
from .dataset import Dataset
from .information import CausalDataInfo


class Histogram(object):

    def __init__(self, conf: ConfigTree):
        hist_conf = conf.get('histogram', conf)
        self.conf = conf
        self.info = CausalDataInfo(conf)
        self.tr_dts = []
        self.max_bin_num = hist_conf.max_bin_num  # Maximum number of bins
        self.min_point_per_bin = hist_conf.min_point_per_bin  # Minimum number of points for binning
        # [leaf, feature, treatment, bin, target]
        self.bin_counts = None
        self.bin_hists = {}
        self._data = None

    def update_hists(self, target, index, leaves_range, treatment, bin_features, is_gradient, is_splitting, threads):
        """
        Update histograms for all nodes in the same level of a tree

        Arguments:
            target (_type_): _description_
            index (_type_): _description_
            leaves_range (_type_): _description_
            treatment (_type_): _description_
            bin_features (_type_): _description_
            is_gradient (bool): _description_
            is_splitting (bool): _description_
            threads (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        n, m = bin_features.shape
        n_w = self.info.n_treatment
        l = len(leaves_range)
        leaves = list(range(0, l, 2))
        n_bins = self.max_bin_num

        if is_gradient:
            assert isinstance(target, (dict, )), f'target should be a dict!'
            keys = [k for k in target.keys()]
            outs = [np.zeros([l, m, n_bins, n_w, target[k].shape[1]], target[k].dtype) for k in keys]
            targets = [target[k] for k in keys]
            # update histogram of target
            update_histograms(targets, bin_features, index, leaves_range, treatment, outs, leaves, n_w, n_bins, threads)
            for i, k in enumerate(keys):
                if l > 1:
                    outs[i][1::2] = self.bin_hists[k][is_splitting] - outs[i][::2]
                self.bin_hists[k] = outs[i]
        else:
            assert isinstance(target, (dict, )), ''
            keys = target.keys()
            outs = [np.zeros([l, m, n_bins, n_w, target[k].shape[1]], target[k].dtype) for k in keys]
            targets = [target[k] for k in keys]
            # update histogram of target
            update_histograms(targets, bin_features, index, leaves_range, treatment, outs, leaves, n_w, n_bins, threads)
            for i, k in enumerate(keys):
                if l > 1:
                    outs[i][1::2] = self.bin_hists[k][is_splitting] - outs[i][::2]
                self.bin_hists[k] = outs[i]
        # update counts
        out = np.zeros([l, m, n_bins, n_w, 1], np.int32)
        update_histogram(np.ones([n, 1], np.int32), bin_features, index, leaves_range, treatment, out, leaves, n_w,
                         n_bins, threads)
        if l > 1:
            out[1::2] = np.expand_dims(self.bin_counts[is_splitting], -1) - out[::2]
        self.bin_counts = out[:, :, :, :, 0]
        return self

    def __getattr__(self, __name: str):
        """
        Get the attribute value.

        Arguments:
            __name (str): The name of the attribute.

        Returns:
            ndarray: The attribute value.

        Raises:
            AttributeError: If the attribute is not found.

        """
        if __name in self.bin_hists:
            return self.bin_hists[__name]
        raise AttributeError()

    @classmethod
    def new_instance(cls, dataset: Dataset, conf: ConfigTree = None, **kwargs):
        """
        Create a new instance of the histogram. 

        Arguments:
            dataset (Dataset): The dataset.
            conf (ConfigTree): The configuration tree.
            kwargs: Additional keyword arguments.

        Returns:
            Histogram: The new instance of the histogram.

        """
        hist = cls(conf, dataset.treatment, dataset.targets)
        hist.binning(dataset)
        return hist
