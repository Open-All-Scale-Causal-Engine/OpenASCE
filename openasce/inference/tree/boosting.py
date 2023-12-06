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

from .bin import BinMapper
from .dataset import Dataset
from .information import CausalDataInfo
from .cppnode import predict
from .utils import to_row_major, INFO, set_log_level_cpp
from .losses import Loss


class Boosting(object):
    def __init__(self, tree_cls, conf: ConfigTree, bin_mapper: BinMapper = None):
        self.info = CausalDataInfo(conf)
        self.conf = conf
        self.n_estimators = conf.n_estimators
        self.trees = []
        self.subfeature = conf.feature_ratio
        self.subsample = conf.instance_ratio
        self.learning_rate = conf.learning_rate
        self.valid_losses = []
        self.bin_mapper = bin_mapper if bin_mapper is not None else BinMapper(conf)
        self.verbose = conf.verbose
        self.print_intervels = conf.get('print_intervels', 5)
        self.tree_cls = tree_cls
        self.n_iter_no_change = conf.n_iter_no_change
        level = conf.get("loglevel", None)
        if level is not None:
            set_log_level_cpp(level)
        self.tol = conf.tol

    def fit(self, data: Dataset):
        """
        Fit the causal forest model on the provided dataset.

        Arguments:
            data: Dataset object containing the input features, targets, and treatment.

        """
        parametrs = self.preprocess(data)

        print_intervels = self.print_intervels

        for i in range(self.n_estimators):
            if i % print_intervels == 0 and self.verbose is True:
                INFO(f'{"="*20}{i}-th tree{"="*20}')

            tree = self.tree_cls(self.conf, self.bin_mapper, verbose=self.verbose)
            g, h = tree.gradients(parametrs['target'], parametrs['pred'])
            cg, ch = tree.gradients(parametrs['target'], parametrs['cpred'])
            tr_data, _, idx, val_idx = self.tr_val(data)
            tree.fit((g[idx], h[idx]), (cg[idx], ch[idx]), tr_data, parametrs['eta'][idx])

            # update current prediction
            self._update_paramers(tree=tree, **parametrs)

            # validation loss
            preloss, postloss = self._validation(
                parametrs['target'][val_idx], parametrs['pred'][val_idx], parametrs['cpred'][val_idx]
            )
            if i % self.print_intervels == 0 and self.verbose is True and len(self.valid_losses) > 0:
                INFO(f'Debias loss:{preloss:.3f}\tpostloss:{postloss:.3f}\ttotal:{self.valid_losses[-1]:.3f}')
            self.trees.append(tree)
            if self.early_stopping():
                if self.verbose is True:
                    INFO(f'early stop')
                break

        self.postprocess()

    def preprocess(self, data: Dataset):
        """
        Perform preprocessing steps on the provided dataset.

        Arguments:
            data: Dataset object containing the input features, targets, and treatment.

        """
        if self.info.feature_columns is None:
            self.info.feature_columns = data.feature_columns
            self.conf.put('dataset.feature', data.feature_columns)
        # check data
        self.check_data(data)
        self.bin_mapper.fit_dataset(data)
        # parameters
        parameters = {}
        parameters['pred'] = np.full_like(data.targets, self.conf.init)
        parameters['cpred'] = np.full_like(data.targets, self.conf.init)

        parameters['target'] = to_row_major(data.targets)
        parameters['features'] = to_row_major(data.features)
        parameters['treatment'] = to_row_major(data.treatment, np.int32)

        parameters['eta'] = np.zeros([data.targets.shape[0], self.info.n_treatment - 1])  # only binary is supported!
        cur_pred = np.zeros_like(data.targets, dtype=parameters['target'].dtype)
        cur_cpred = np.zeros_like(data.targets, dtype=parameters['target'].dtype)
        parameters['out'] = (cur_pred, cur_cpred)

        return parameters

    def check_data(self, data: Dataset):
        n = data.features.shape[0]
        assert (n == data.targets.shape[0]) and (
            data.treatment.shape[0] == n
        ), f'You should guarantee the number of features, outcomes and treatment are the same!'

        assert (
            data.targets.shape[1] == self.info.n_period
        ), f'The number of outcomes({data.targets.shape[1]}) does\'t equals to {self.info.n_period}!'

        assert self.info.n_period > self.info.treat_dt, f'treat_dt should less than n_period!'

    def tr_val(self, data: Dataset, subsample=None, **kwargs):
        """
        Split the dataset into training and validation sets.

        Arguments:
            data: Dataset object containing the input features, targets, and treatment.
            subsample: Ratio of instances to include in the training set. If None, uses the instance_ratio from conf.

        Returns:
            Tuple containing the training dataset, validation dataset, indices of training instances, and indices of validation instances.

        """
        n = len(data)
        if subsample is None:
            subsample = self.info.instance_ratio
        if subsample >= 1:
            return data, data, slice(None), slice(None)
        tr_n = int(n * subsample)
        idx = np.random.permutation(n).astype(np.int32)
        return data.sub_dataset(idx[:tr_n]), data.sub_dataset(idx[tr_n:]), idx[:tr_n], idx[tr_n:]

    def postprocess(self):
        opt_n = np.argmin(self.valid_losses)
        self.trees = self.trees[: opt_n + 1]

    def _validation(self, target, prediction, cprediction=None):
        op_loss = Loss.new_instance(self.conf.tree)
        preloss = 0
        if self.info.treat_dt > 0 and cprediction is not None:
            preloss = op_loss.loss(target[:, : self.info.treat_dt], cprediction[:, : self.info.treat_dt]).mean()
        postloss = op_loss.loss(target[:, self.info.treat_dt :], prediction[:, self.info.treat_dt :]).mean()
        self.valid_losses.append(preloss + self.conf.tree.coefficient * postloss)
        return preloss, postloss

    def _update_paramers(self, *args, **kwargs):
        tree = kwargs.pop('tree')
        features = kwargs.pop('features')
        treatment = kwargs.pop('treatment')
        out = kwargs.pop('out')
        pred = kwargs.pop('pred')
        cpred = kwargs.pop('cpred')
        eta = kwargs.pop('eta', None)

        cur_pred, cur_cpred, cur_eta = tree.predict(features, treatment, key='cf_outcomes', out=out)
        pred += self.learning_rate * cur_pred
        cpred += self.learning_rate * cur_cpred
        if cur_eta is not None:
            eta += self.learning_rate * cur_eta[:, :1]
        return {'pred': pred, 'cpred': cpred, 'eta': eta}

    def early_stopping(self) -> bool:
        """
        Check if early stopping criteria is met based on the validation losses.

        Returns:
            True if early stopping criteria is met, False otherwise.

        """
        # use both pre_loss and post_loss
        min_loss = min(self.valid_losses)
        cur_loss = self.valid_losses[-1]
        no_change_steps = self.n_iter_no_change if np.isscalar(self.n_iter_no_change) else np.inf
        n, opt_n = len(self.valid_losses), np.argmin(self.valid_losses)
        if cur_loss > min_loss * (1 + self.tol) and no_change_steps <= n - opt_n - 1:
            return True
        if len(self.trees) > 0 and self.trees[-1].root.is_leaf:
            INFO(f'The last tree is no more splitting!')
            return True
        self.opt_step = opt_n
        if n - opt_n > 1 and self.verbose is True:
            INFO(f'{opt_n}-th has min loss ({min_loss:.3}), no change steps: {n - opt_n - 1}')
        return False

    def predict(self, X, key: str, *, data: Dataset = None):
        """
        Predict the output using the trained model on the input data.

        Arguments:
            X: Feature matrix of the input data.
            key: Type of prediction, can be 'leaf_id', 'effect', or 'effect-ND'.
            data: Dataset object containing the feature data.

        Returns:
            Prediction result based on the specified key.

        Raises:
            RuntimeError: If the specified key is unknown and not supported.

        """
        if X is None:
            X = data.features
        x = to_row_major(X)
        if key == 'leaf_id':
            leaf_ids = np.zeros([x.shape[0], len(self.trees)], dtype=np.float)
            predict([tree.export()[0] for tree in self.trees], x, leaf_ids, 'leaf_id')
            return leaf_ids.astype(int)
        elif key == 'effect':
            results = np.zeros([x.shape[0], len(self.trees), 2, self.info.n_period], dtype=np.float64)
            predict([tree.export()[0] for tree in self.trees], x, results, 'outcomes')
            tau_hat = self.learning_rate * results.sum(axis=1)
            tau_hat = tau_hat[:, 1] - tau_hat[:, 0]
            if self.info.treat_dt > 0:
                tau_hat = tau_hat[:, self.info.treat_dt :] - tau_hat[:, : self.info.treat_dt].mean(
                    axis=1, keepdims=True
                )
            else:
                tau_hat = tau_hat[:, self.info.treat_dt :]
            return tau_hat
        elif key == 'effect-ND':
            results = np.zeros([x.shape[0], len(self.trees), 2, self.info.n_period], dtype=np.float64)
            predict([tree.export()[0] for tree in self.trees], x, results, 'outcomes')
            tau_hat = self.learning_rate * results.sum(axis=1)
            tau_hat = tau_hat[:, 1] - tau_hat[:, 0]
            tau_hat = tau_hat[:, self.info.treat_dt]
            return tau_hat

        raise RuntimeError(f'Only `leaf_id`, `effect` and `effect-ND` are supported, but {key} is unknown!')

    def effect(self, X=None, *, data: Dataset = None):
        """Predict the treatment effect on the input data."""
        return self.predict(X, 'effect', data=data)

    def split_counts(self, trees=None, feature_names=None):
        """
        Count the number of splits made on each feature in the gradient boost causal trees.

        Arguments:
            trees: List of decision trees. If None, uses the trained trees.
            feature_names: List of feature names. If None, uses the feature columns from conf.

        Returns:
            Dictionary with feature names as keys and the corresponding split counts as values.

        """
        if feature_names is None:
            feature_names = self.info.feature_columns
        counts = np.zeros([len(feature_names)], dtype=int)
        if trees is None:
            trees = self.trees
        for tree in trees:
            for node in tree.nodes:
                counts[node.split_feature] += 1
        return {feature_names[i]: cnt for i, cnt in enumerate(counts)}
