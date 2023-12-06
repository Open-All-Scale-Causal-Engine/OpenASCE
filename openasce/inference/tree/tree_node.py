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

from .losses import Loss
from .utils import groupby


class CausalTreeNode(object):
    """A class for a node in a Causal Tree maximizing heterogenous treatment effect."""
    def __init__(self, conf: ConfigTree = None, **kwargs):
        conf_tree = conf.get('tree', conf)
        self.leaf_id: int = kwargs.get('leaf_id')
        self.level_id: int = kwargs.get('level_id')

        self.n_period = conf.get('dataset.n_period')
        self.treat_dt = conf.get('dataset.treat_dt')
        self.n_treatment = conf.get('dataset.n_treatment')

        self.is_leaf = False
        loss_op = Loss.new_instance(conf_tree)
        self.conf = conf
        self.op_loss: Loss = loss_op
        self.min_point_num_node = conf_tree.min_point_num_node  # Minimum number of points for node splitting
        self.max_depth = conf_tree.max_depth

        self.lambd = kwargs.get('lambd', conf.get('tree.lambd'))
        self.coef = kwargs.get('coef', conf.get('tree.coefficient'))
        self.ps_weight = True
        self.gain_thresh = 0
        self._children = []

        self.split_feature = None
        self.split_thresh = None
        self.split_rawthresh = None
        self.theta = kwargs.get('theta', None)
        self._effect = None
        self._estimator = None

    def estimate(self, outcome: np.ndarray, treatment: np.ndarray, weight: np.ndarray = None):
        """
        Estimate the treatment effect given the outcome, treatment, and weight.

        Arguments:
            outcome: The outcome values.
            treatment: The treatment values.
            weight: The weight values.

        Returns:
            ndarray: The estimated treatment effect.

        """
        # by simple mean
        if weight is None:
            weight = 1
        outcome *= weight
        out = groupby(outcome, treatment, 'mean')
        return np.stack([out[i] for i in range(len(out))], axis=0)

    def estimate_by_hist(self, outcome: np.ndarray, treatment: np.ndarray, count: np.ndarray):
        """
        Estimate the treatment effect using histogram-based method.

        Arguments:
            outcome: The outcome values.
            treatment: The treatment values.
            count: The count values.

        Returns:
            ndarray: The estimated treatment effect.

        """
        return outcome / count

    @property
    def children(self):
        return self._children

    def effect(self, w):
        """
        Compute the treatment effect.

        Arguments:
            w: The treatment weights.

        Returns:
            ndarray: The computed treatment effect.

        """
        if self._effect is None:
            if self.theta is not None:
                return self.theta[w] - self.theta[0]
            return None
        return self._effect


class GradientCausalTreeNode(CausalTreeNode):
    """A class for a node in a Gradient Boosting Causal Tree."""
    def __init__(self, conf: ConfigTree = None, **kwargs):
        super().__init__(conf, **kwargs)
        self.eta = kwargs.get('eta', None)

    def estimate(self, G, H, **kwargs):
        """
        Estimate the treatment effect given the gradients and hessians.

        Arguments:
            G: The gradients.
            H: The hessians.
            **kwargs: Additional keyword arguments.

        Returns:
            ndarray: The estimated treatment effect.

        """
        lambd = kwargs.get('lambd', 0)
        return -G / (H + lambd)
