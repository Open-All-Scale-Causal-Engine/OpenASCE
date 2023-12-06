#    Copyright 2023 AntGroup CO., Ltd.
#
#    Licensed under the Apache License, Version 2.0 (the 'License');
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an 'AS IS' BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import random

import numpy as np
from pyhocon import ConfigTree, ConfigFactory

from .bin import BinMapper
from .dataset import Dataset, PsudoDataset
from .boosting import Boosting
from .gradient_causal_tree import GradientDiDCausalTree
from .utils import to_row_major
from .cppnode import predict


class DifferenceInDifferencesRegressionTree(Boosting):
    """
    Gradient Boosting debiased Causal Tree for regression,
    Difference-in-differences meets tree-based methods: heterogeneous treatment effects estimation with unmeasured confounding

    Reference:
    1. Tang, C., Wang, H., Li, X., Qing, C., Li, L., & Zhou, J. (2023). Difference-in-Differences Meets Tree-Based Methods: Heterogeneous Treatment Effects Estimation with Unmeasured Confounding. Proceedings of the 40th International Conference on Machine Learning.


    Arguments:
        n_estimators : int, default=100
            The number of boosting stages to perform. Gradient boosting
            is fairly robust to over-fitting so a large number usually
            results in better performance.

        learning_rate : float, default=0.1
            learning rate shrinks the contribution of each tree by `learning_rate`.
            There is a trade-off between learning_rate and n_estimators.

        subsample : float, default=1.0
            The fraction of samples to be used for fitting the individual base
            learners. If smaller than 1.0 this results in Stochastic Gradient
            Boosting. `subsample` interacts with the parameter `n_estimators`.
            Choosing `subsample < 1.0` leads to a reduction of variance
            and an increase in bias.

        subfeature : float, default=1.0
            The fraction of feature to be used for fitting the individual base
            learners. Referring to random forest.

        train_loss : string, default=openasce.inference.tree.losses.MeanSquaredError.
            The spliting criterion used by GBCT. User can reimplement by inheriting base class `Loss`.

        min_samples_split : int, default=10
            The minimum number of samples required to split an internal node

        max_depth : int, default=3
            maximum depth of the individual regression estimators. The maximum
            depth limits the number of nodes in the tree. Tune this parameter
            for best performance; the best value depends on the interaction
            of the input variables.

        lambd : float, default=5
            The regularization parameter of l2_penalty reference to XGB.

        coef : float, default=1
            The regularization parameter of selection bias, which refer to GBCT.

        parallel_l2: float, default=0

        n_period (int, optional): The number of timesteps. It's required to be provied by user.

        treat_dt (int, optional): The time step that treatment is assigned, which is less than n_period. It's
            required to be provied by user.

        init : The initial prediction. Default to 0.

        random_state : int or RandomState, default=None
            Controls the random seed given to each Tree estimator at each
            boosting iteration.
            In addition, it controls the random permutation of the features at
            each split (see Notes for more details).

        verbose : bool, default=False. Enable verbose output.

        n_iter_no_change : int, default=40
            ``n_iter_no_change`` is used to decide if early stopping will be used
            to terminate training when validation score is not improving. By
            default it is set to None to disable early stopping.

        tol : float, default=1e-4
            Tolerance for the early stopping. When the loss is not improving
            by at least tol for ``n_iter_no_change`` iterations (if set to a
            number), the training stops.

        nthreads: The number of threads to use for parallelization. Default is 32.

        conf (ConfigTree, optional): _description_. Defaults to None.

        bin_mapper (BinMapper, optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        subsample=1.0,
        subfeaure=1.0,
        train_loss="openasce.inference.tree.losses.MeanSquaredError",
        min_samples_split=10,
        max_depth=3,
        lambd=5,
        coeff=1,
        parallel_l2=0,
        n_period=None,
        treat_dt=None,
        init=0,
        random_state=None,
        verbose=False,
        n_iter_no_change=10,
        tol=1e-4,
        nthreads=32,
        *,
        conf: ConfigTree = None,
        bin_mapper: BinMapper = None,
    ):
        if conf is None:
            conf = ConfigFactory.from_dict({})
        if isinstance(conf, dict):
            conf = ConfigFactory.from_dict(conf)

        conf.put("learning_rate", conf.get("learning_rate", learning_rate))
        conf.put("n_estimators", conf.get("n_estimators", n_estimators))
        conf.put("feature_ratio", conf.get("feature_ratio", subfeaure))
        conf.put("instance_ratio", conf.get("instance_ratio", subsample))
        conf.put("task_type", "regression")

        conf.put("init", conf.get("init", init))
        conf.put("random_state", conf.get("tree.random_state", random_state))
        conf.put("verbose", conf.get("verbose", verbose))
        conf.put("n_iter_no_change", conf.get("n_iter_no_change", n_iter_no_change))
        conf.put("tol", conf.get("tol", tol))
        conf.put("nthreads", conf.get("nthreads", nthreads))

        conf.put(
            "tree.min_point_num_node",
            conf.get("tree.min_point_num_node", min_samples_split),
        )
        conf.put("tree.max_depth", conf.get("tree.max_depth", max_depth))
        conf.put("tree.lambd", conf.get("tree.lambd", lambd))
        conf.put("tree.parallel_l2", conf.get("tree.parallel_l2", parallel_l2))
        conf.put("tree.coefficient", conf.get("tree.coefficient", coeff))
        conf.put("tree.loss_cls", conf.get("tree.train_loss", train_loss))

        conf.put("dataset.n_treatment", 2)
        conf.put("dataset.n_period", conf.get("dataset.n_period", n_period))
        conf.put("dataset.treat_dt", conf.get("dataset.treat_dt", treat_dt))

        conf.put("histogram.max_bin_num", conf.get("histogram.max_bin_num", 64))
        conf.put(
            "histogram.min_point_per_bin", conf.get("histogram.min_point_per_bin", 10)
        )

        np.random.seed(random_state)
        random.seed(random_state)

        super().__init__(GradientDiDCausalTree, conf, bin_mapper)

    def fit(
        self,
        X: np.ndarray = None,
        Y: np.ndarray = None,
        D: np.ndarray = None,
        *,
        data: Dataset = None,
        data_test: Dataset = None,
    ):
        """
        train the GradientBoostingUpliftTree

        Arguments:
            X: np.ndarray, [n_instances, n_features].   features
            Y: np.ndarray, [n_instances, n_period].    outcome
            D: np.ndarray, [n_instances,].               treatment.
            data (Dataset, optional): _description_. Defaults to None.

        Returns:
        """
        if X is not None and Y is not None and D is not None:
            if len(Y.shape) == 1:
                Y = np.expand_dims(Y, axis=1)
                D = D.astype(np.int32)
            data = PsudoDataset(X, Y, D)
        return super().fit(data=data)

    def effect(self, X: np.ndarray = None, *, data: Dataset = None):
        """
        predict the treatment effect.

        Arguments:
            X: np.ndarray, [n_instances, n_features].   features
            data: Dataset, optional. Defaults to None.

        Returns:
            np.ndarray, [n_instances,]. treatment effect.
        """
        assert (X is not None) or (
            data is not None
        ), f"One of `X` and `data` must be provided!"
        return super().effect(X, data=data)

    def predict(
        self, X: np.ndarray = None, key: str = "effect", *, data: Dataset = None
    ):
        """
        predict the treatment effect or leaf id, which is determined by the parameter `key`.

        Arguments:
            X (np.ndarray): features
            key (str): the key can be 'effect', 'leaf_id' or 'outcome'.
            data (Dataset, optional): dataset. Defaults to None.

        Returns:
            _type_: _description_
        """
        assert (X is not None) or (
            data is not None
        ), f"One of `X` and `data` must be provided!"
        if X is None:
            X = data.features
        x = to_row_major(X)
        if key in ("leaf_id", "outcome"):
            return super().predict(X, key, data=data)
        elif key == "effect":
            shape = (x.shape[0], len(self.trees), self.info.n_period)
            tau_hat = np.zeros(shape)
            predict(
                [tree.export()[0] for tree in self.trees], x, tau_hat, "debiased_effect"
            )
            tau_hat = self.learning_rate * tau_hat.sum(axis=1)
            return tau_hat[:, self.info.treat_dt :]
        elif key == "eta":
            shape = (x.shape[0], len(self.trees))
            eta = np.zeros(shape)
            predict([tree.export()[0] for tree in self.trees], x, eta, "eta")
            return eta

        raise RuntimeError(f"Unkown key={key}")
