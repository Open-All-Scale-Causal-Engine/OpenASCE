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

from typing import Dict, List
from collections import OrderedDict

from pyhocon import ConfigTree
import numpy as np

from .tree_node import GradientCausalTreeNode
from .dataset import Dataset
from .histogram import Histogram
from .information import CausalDataInfo
from .bin import BinMapper
from .splitting_losses import *
from .cppnode import create_didnode_from_dict, predict
from .losses import Loss
from .utils import *


def _filter(leaves, leaves_range):
    inner_leaves_range = []
    inner_leaves = []
    splitting = []
    for i, leaf in enumerate(leaves):
        if leaf.is_leaf is False:
            inner_leaves_range.append(leaves_range[i])
            inner_leaves.append(leaf)
            splitting.append(True)
        else:
            splitting.append(False)
    if len(inner_leaves_range) > 0:
        inner_leaves_range = np.stack(inner_leaves_range, axis=0)
    else:
        return [], None, splitting
    return inner_leaves, inner_leaves_range, splitting


class GradientDebiasedCausalTree:
    """
    GradientDebiasedCausalTree is a class that represents a gradient-based debiased causal tree model.

    Arguments:
        conf (ConfigTree): The configuration tree.
        bin_mapper (BinMapper): The BinMapper instance.
        kwargs: Additional keyword arguments.

    """

    def __init__(self, conf: ConfigTree = None, bin_mapper: BinMapper = None, **kwargs):
        self.conf = conf
        self.info = CausalDataInfo(conf)
        self.verbose = kwargs.get('verbose', False)
        self.feature_used = []
        self.feature_used_map = {}  # key: sub-feature index, value: original feature index
        self.bin_mapper = bin_mapper
        conf_tree = conf.get('tree', conf)
        self.op_loss = Loss.new_instance(conf_tree)
        self.w_monotonic = 0
        self.did = False
        self.nthreads = kwargs.get('nthreads', conf.get('nthreads', 32))

    def fit(self, gradients, cgradients, data: Dataset, eta=None):
        """
        Fit the GradientDebiasedCausalTree model.

        Arguments:
            gradients: The gradients.
            cgradients: The counterfactal gradients.
            data (Dataset): The training dataset.
            eta: The eta values.

        Returns:
            None

        """
        hist, idx_map = self.preprocess(gradients, cgradients, data, eta, 1, self.info.feature_ratio)
        root = GradientCausalTreeNode(self.conf, leaf_id=0, level_id=0)
        self.root = root
        leaves = [root]
        leaves_range = np.array([[0, self.inst_used]], np.int32)
        # calculate loss
        leaf_id = root.leaf_id
        gsum = hist.bin_grad_hist[leaf_id, 0].sum(0)
        hsum = hist.bin_hess_hist[leaf_id, 0].sum(0)
        root.theta = root.estimate(gsum, hsum, lambd=self.info.lambd)
        if eta is not None:
            self.root.eta = np.zeros([self.info.n_treatment, 1])
        for i in range(self.info.max_depth):
            if self.verbose:
                TRACE(f'{"--"*10} the {i}-th iterations {"--"*10}')
            split_conds = self.split(leaves, hist)
            leaves, leaves_range = self.updater(
                split_conds, gradients, cgradients, data, hist, idx_map, leaves, leaves_range, eta
            )
            if len(leaves) == 0 and self.verbose:
                TRACE(f'{i}-th level early stop!')
                break
        for leaf in leaves:
            leaf.is_leaf = True
        self.postprocess()

    def updater(
        self,
        split_conds: Dict,
        gradients,
        cgradients,
        tr_data,
        hist: Histogram,
        idx_map,
        leaves: List[GradientCausalTreeNode],
        leaves_range,
        eta=None,
    ):
        """
        Update the tree nodes.

        Arguments:
            split_conds (Dict): The split conditions.
            gradients: The gradients.
            cgradients: The cgradients.
            tr_data (Dataset): The training dataset.
            hist (Histogram): The histogram.
            idx_map: The index map.
            leaves (List[GradientCausalTreeNode]): The list of tree nodes.
            leaves_range: The range of each leaf.
            eta: The eta values.

        Returns:
            Tuple[List[GradientCausalTreeNode], ndarray]: The updated tree nodes and updated leaf ranges.

        """
        leaves, leaves_range, is_splitting = _filter(leaves, leaves_range)
        n_leaf = len(split_conds)
        if len(leaves) == 0 or len(split_conds) == 0:
            return leaves, leaves_range
        x_binned = to_row_major(tr_data.bin_features[self.feature_used], np.int32)
        treatment = to_row_major(tr_data.treatment, np.int32)
        sorted_split = OrderedDict(sorted(split_conds.items()))
        split_info = np.asfarray([[info['feature'], info['threshold']] for _, info in sorted_split.items()]).astype(
            np.int32
        )
        out = np.zeros([n_leaf * 2, 2], np.int32)
        update_x_map(x_binned, idx_map, split_info, leaves_range, out)
        leaves_range_new = out
        hist.update_hists(
            {
                'bin_grad_hist': gradients[0],
                'bin_hess_hist': gradients[1],
                'bin_cgrad_hist': cgradients[0],
                'bin_chess_hist': cgradients[1],
            },
            idx_map,
            leaves_range_new,
            treatment,
            x_binned,
            is_gradient=True,
            is_splitting=is_splitting,
            threads=self.nthreads,
        )
        # create new node
        leaves_new = []
        for i, leaf in enumerate(leaves):
            ltheta, rtheta = split_conds[leaf.level_id]['theta']
            l_eta, r_eta = split_conds[leaf.level_id]['eta']
            leaf._children = [
                GradientCausalTreeNode(
                    self.conf, leaf_id=leaf.leaf_id * 2 + 1, level_id=i * 2, theta=ltheta, eta=l_eta
                ),
                GradientCausalTreeNode(
                    self.conf, leaf_id=leaf.leaf_id * 2 + 2, level_id=i * 2 + 1, theta=rtheta, eta=r_eta
                ),
            ]
            leaves_new.extend(leaf._children)
            fid, bin_id = split_info[i]
            leaf.split_feature = self.feature_used_map[fid]
            leaf.split_thresh = bin_id
            leaf.split_rawthresh = self.bin_mapper.inverse_transform(bin_id, self.feature_used_map[fid])
        return leaves_new, leaves_range_new

    def split(self, leaves: List[GradientCausalTreeNode], hist: Histogram):
        """
        Split the tree nodes.

        Arguments:
            leaves (List[GradientCausalTreeNode]): The list of tree nodes.
            hist (Histogram): The histogram.

        Returns:
            Dict: The split conditions.

        """
        return self._split_cpp(leaves, hist)

    def _split_cpp(self, leaves: List[GradientCausalTreeNode], hist: Histogram):
        """
        Split the tree nodes using C++ implementation.

        Arguments:
            leaves (List[GradientCausalTreeNode]): The list of tree nodes.
            hist (Histogram): The histogram.

        Returns:
            Dict: The split conditions.

        """
        info = self.info
        n_leaves, m, n_bins, n_w, n_y = hist.bin_grad_hist.shape
        t0, T, n_w = info.treat_dt, info.n_period, info.n_treatment
        lambd = info.lambd
        coef = info.coef
        min_num = self.info.min_point_num_node
        configs = {leaf.level_id: {fid: [0, n_bins] for fid in range(m)} for leaf in leaves}
        parameters = f"""{{
            "tree": {{
                "lambd": {lambd},
                "coeff": {coef},
                "min_point_num_node": {min_num},
                "min_var_rate": {0.1},
                "monotonic_constraints": {self.w_monotonic}
            }},
            
            "threads": {self.nthreads},
            "dataset": {{
                "treat_dt": {t0}
            }}
        }}"""

        res = gbct_splitting_losses(
            configs,
            hist.bin_grad_hist,
            hist.bin_hess_hist,
            hist.bin_cgrad_hist,
            hist.bin_chess_hist,
            hist.bin_counts,
            parameters,
        )
        split_conds = {}
        for leaf in leaves:
            level_id = leaf.level_id
            if level_id not in res:
                leaf.is_leaf = True
                continue
            # opt_feature, opt_bin_idx, opt_loss
            if bool(np.isinf(res[level_id][2])) is False:
                split_conds[level_id] = {
                    'feature': res[level_id][0],
                    'threshold': res[level_id][1],
                    'loss': res[level_id][2],
                    'theta': (res[level_id][3][0], res[level_id][3][1]),
                    'eta': (None, None),
                }
            else:
                leaf.is_leaf = True
        return split_conds

    def preprocess(self, gradients, cgradients, tr_data: Dataset, eta=None, subsample=1, subfeature=1):
        """
        Preprocess the data before fitting the tree.

        Arguments:
            gradients: The gradients.
            cgradients: The cgradients.
            tr_data (Dataset): The training dataset.
            eta: The eta values.
            subsample: The subsample ratio.
            subfeature: The subfeature ratio.

        Returns:
            Tuple[Histogram, ndarray]: The histogram and index.

        """
        n, m = tr_data.features.shape
        index = np.random.permutation(n).astype(np.int32)
        n_used, m_used = np.math.ceil(n * subsample), np.math.ceil(m * subfeature)

        features = self.info.feature_columns
        # subsampling
        if m_used < m:
            tmp_feat = np.random.choice(self.info.feature_columns, m_used, replace=False)
            features = [f for f in self.info.feature_columns if (f in tmp_feat)]
        else:
            features = self.info.feature_columns
        hist = Histogram(self.conf)
        if tr_data.bin_features is None:
            self.bin_mapper.fit_dataset(tr_data)
        else:
            hist.columns = list(tr_data.feature_columns)
        x_binned = to_row_major(tr_data.bin_features[features], np.int32)
        self.feature_used = features
        self.inst_used = n_used
        orig_features = list(tr_data.feature_columns)
        self.feature_used_map = {i: orig_features.index(f) for i, f in enumerate(features)}
        # calculate histogram for outcome
        w = to_row_major(tr_data.treatment, np.int32)
        hist.update_hists(
            {
                'bin_grad_hist': gradients[0],
                'bin_hess_hist': gradients[1],
                'bin_cgrad_hist': cgradients[0],
                'bin_chess_hist': cgradients[1],
            },
            index,
            np.array([[0, n_used]], np.int32),
            w,
            x_binned,
            True,
            [],
            self.nthreads,
        )
        return hist, index

    def export(self):
        """
        Export the tree model.

        Returns:
            Tuple[List[DidNode], List[Dict]]: The exported C++ nodes and Python nodes.

        """
        nodes, queue = [], [self.root]
        while len(queue) > 0:
            nodes.append(queue.pop(0))
            for child in nodes[-1].children:
                queue.append(child)
        # encode for each node
        slim_cppnodes, slim_nodes = [], []
        t_0 = self.info.treat_dt
        for child in nodes:
            bias, effect, debiased_effect = [], [], []
            for _w in range(1, self.info.n_treatment):
                if t_0 > 0:
                    bias.append(np.mean(child.theta[_w, :t_0] - child.theta[0, :t_0]))
                else:
                    bias.append(0)
                effect.append(child.theta[_w] - child.theta[0])
                if isinstance(self, GradientDebiasedCausalTree):
                    debiased_effect.append(effect[-1][t_0:] - bias[-1])
                else:
                    debiased_effect.append(effect[t_0:] + child.eta[_w])

            info = {
                'leaf_id': child.leaf_id,
                'level_id': child.level_id,
                'outcomes': child.theta,
                'predict': child.theta,
                'bias': np.array(bias),
                'eta': child.eta,
                'effect': np.array(effect),
                'debiased_effect': np.array(debiased_effect),
                'is_leaf': child.is_leaf,
                'children': [-1, -1],
                'split_feature': -1,
                'split_thresh': -1,
            }
            if child.is_leaf is False:
                info['children'] = [nodes.index(child.children[0]), nodes.index(child.children[1])]
                info['split_feature'] = child.split_feature
                info['split_thresh'] = child.split_rawthresh
            slim_cppnodes.append(create_didnode_from_dict(info))
            info['split_feature'] = self.info.feature_columns[info['split_feature']]
            slim_nodes.append(info)
        return slim_cppnodes, slim_nodes

    def postprocess(self):
        """
        Perform post-processing steps after fitting the tree.

        Returns:
            None

        """
        nodes, queue = [], [self.root]
        while len(queue) > 0:
            nodes.append(queue.pop(0))
            for child in nodes[-1].children:
                queue.append(child)
        self.nodes = nodes

    def _predict(self, nodes, x, key='effect', out=None):
        """
        Internal method to predict using the tree nodes.

        Arguments:
            nodes: The tree nodes.
            x: The input data.
            key: The prediction key.
            out: The output array.

        Returns:
            ndarray: The predicted values.

        Raises:
            NotImplementedError: If the prediction key is not implemented.

        """
        assert isinstance(nodes, list) and len(nodes) > 0, f'nodes must be list and at least one element!'
        if key == 'effect':
            shape = (x.shape[0],) + nodes[0].outcomes.shape
        elif key == 'leaf_id':
            shape = (x.shape[0], 1, 1)
        elif key == 'outcomes':
            shape = (x.shape[0],) + nodes[0].outcomes.shape
        elif key == 'eta':
            shape = (x.shape[0], 2, 1)
        else:
            raise NotImplementedError
        if x.flags.c_contiguous is False:
            x = to_row_major(x)
        if out is None:
            out = np.zeros(shape, np.float64)
        predict(nodes, x, out, key)
        return out

    def predict(self, x, w=None, key='effect', out=None):
        """
        Predict the treatment effect or other values.

        Arguments:
            x: The input data.
            w: The treatment weights.
            key: The prediction key.
            out: The output array.

        Returns:
            ndarray: The predicted values.

        """
        if key == 'cf_outcomes':
            outcome = self.predict(x, key='outcomes')
            cur_pred, cur_cpred = out
            indexbyarray(outcome, w, cur_pred, cur_cpred)
            return cur_pred, cur_cpred, None
        return self._predict(self.export()[0], x, key, out)

    def gradients(self, target, prediction, **kwargs):
        """
        Compute the gradients of the loss function.

        Arguments:
            target: The target values.
            prediction: The predicted values.
            kwargs: Additional keyword arguments.

        Returns:
            ndarray: The gradients.

        """
        return self.op_loss.gradients(target, prediction, **kwargs)

    def loss(self, grad, hess, y_hat=None, **kwargs):
        """
        Compute the loss function.

        Arguments:
            grad: The gradients.
            hess: The hessians.
            y_hat: The predicted values.
            kwargs: Additional keyword arguments.

        Returns:
            ndarray: The loss values.

        """
        lambd = kwargs.pop('lambd', 0)
        return self.root.fast_loss(y_hat=y_hat, grad=grad, hess=hess, lambd=lambd, **kwargs)


class GradientDiDCausalTree(GradientDebiasedCausalTree):
    """
    GradientDiDCausalTree is a class that represents a gradient-based debiased causal tree model with difference in
    differences. It inherits from the GradientDebiasedCausalTree class.

    Arguments:
        conf (ConfigTree): The configuration tree.
        bin_mapper (BinMapper): The BinMapper instance.
        kwargs: Additional keyword arguments.
    """

    def preprocess(self, gradients, cgradients, tr_data: Dataset, eta=None, subsample=1, subfeature=1):
        """
        Preprocesses the data for the GradientDiDCausalTree model.

        Arguments:
            gradients: The gradients.
            cgradients: The counterfactual gradients.
            tr_data (Dataset): The training dataset.
            eta: The parallel interval between the treated and control group.
            subsample (float): The subsampling ratio for instances (default: 1).
            subfeature (float): The subsampling ratio for features (default: 1).

        Returns:
            hist (Histogram): The constructed histogram.
            index (ndarray): The permutation index.
        """
        n, m = tr_data.features.shape
        index = np.random.permutation(n).astype(np.int32)
        n_used, m_used = np.math.ceil(n * subsample), np.math.ceil(m * subfeature)

        features = self.info.feature_columns
        # subsampling
        if m_used < m:
            tmp_feat = np.random.choice(self.info.feature_columns, m_used, replace=False)
            features = [f for f in self.info.feature_columns if (f in tmp_feat)]
        else:
            features = self.info.feature_columns
        if self.verbose:
            INFO(f'=' * 50)
            INFO(f'{"=" * 5} Used instance & features: {n_used}, {m_used}, {"=" * 5}')
            INFO(f'=' * 50)
        hist = Histogram(self.conf)
        if tr_data.bin_features is None:
            self.bin_mapper.fit_dataset(tr_data)
        else:
            hist.columns = list(tr_data.feature_columns)
        x_binned = to_row_major(tr_data.bin_features[features], np.int32)
        self.feature_used = features
        self.inst_used = n_used
        orig_features = list(tr_data.feature_columns)
        self.feature_used_map = {i: orig_features.index(f) for i, f in enumerate(features)}
        # calculate histogram for outcome
        w = to_row_major(tr_data.treatment, np.int32)
        hist.update_hists(
            {
                'bin_grad_hist': gradients[0],
                'bin_hess_hist': gradients[1],
                'bin_cgrad_hist': cgradients[0],
                'bin_chess_hist': cgradients[1],
                'bin_eta_hist': eta,
            },
            index,
            np.array([[0, n_used]], np.int32),
            w,
            x_binned,
            True,
            [],
            self.nthreads,
        )
        return hist, index

    def updater(
        self,
        split_conds: Dict,
        gradients,
        cgradients,
        tr_data,
        hist: Histogram,
        idx_map,
        leaves: List[GradientCausalTreeNode],
        leaves_range,
        eta,
    ):
        """
        Update the GradientCausalTree by performing splitting and updating histograms.

        Arguments:
            split_conds (Dict): The split conditions.
            gradients: The gradients.
            cgradients: The counterfactual gradients.
            tr_data: The training dataset.
            hist (Histogram): The histogram object.
            idx_map: The index mapping.
            leaves (List[GradientCausalTreeNode]): The list of leaves.
            leaves_range: The range of leaves.
            eta: The parallel interval between the treated and control group.

        Returns:
            leaves_new (List[GradientCausalTreeNode]): The new leaves.
            leaves_range_new: The new range of leaves.
        """
        leaves, leaves_range, is_splitting = _filter(leaves, leaves_range)
        n_leaf = len(split_conds)
        if len(leaves) == 0 or len(split_conds) == 0:
            return leaves, leaves_range
        x_binned = to_row_major(tr_data.bin_features[self.feature_used], np.int32)
        treatment = to_row_major(tr_data.treatment, np.int32)
        sorted_split = OrderedDict(sorted(split_conds.items()))
        split_info = np.asarray([[info['feature'], info['threshold']] for _, info in sorted_split.items()]).astype(
            np.int32
        )
        out = np.zeros([n_leaf * 2, 2], np.int32)
        update_x_map(x_binned, idx_map, split_info, leaves_range, out)
        leaves_range_new = out
        hist.update_hists(
            {
                'bin_grad_hist': gradients[0],
                'bin_hess_hist': gradients[1],
                'bin_cgrad_hist': cgradients[0],
                'bin_chess_hist': cgradients[1],
                'bin_eta_hist': eta,
            },
            idx_map,
            leaves_range_new,
            treatment,
            x_binned,
            is_gradient=True,
            is_splitting=is_splitting,
            threads=self.nthreads,
        )
        # create new node
        leaves_new = []
        for i, leaf in enumerate(leaves):
            ltheta, rtheta = split_conds[leaf.level_id]['theta']
            l_eta, r_eta = split_conds[leaf.level_id]['eta']
            leaf._children = [
                GradientCausalTreeNode(
                    self.conf, leaf_id=leaf.leaf_id * 2 + 1, level_id=i * 2, theta=ltheta, eta=l_eta
                ),
                GradientCausalTreeNode(
                    self.conf, leaf_id=leaf.leaf_id * 2 + 2, level_id=i * 2 + 1, theta=rtheta, eta=r_eta
                ),
            ]
            leaves_new.extend(leaf._children)
            fid, bin_id = split_info[i]
            leaf.split_feature = self.feature_used_map[fid]
            leaf.split_thresh = bin_id
            leaf.split_rawthresh = self.bin_mapper.inverse_transform(bin_id, self.feature_used_map[fid])
        return leaves_new, leaves_range_new

    def _split_cpp(self, leaves: List[GradientCausalTreeNode], hist: Histogram):
        """
        Split the leaf nodes at the current level using C++ implementation.

        Arguments:
            leaves: The list of leaf nodes.
            hist: The histogram object.

        Returns:
            split_conds: The split conditions.
        """
        # Step 1: Collect all the split points that need to calculate the loss
        # Step 2: Perform the split
        info = self.info
        n_leaves, m, n_bins, n_w, n_y = hist.bin_grad_hist.shape
        t0, T, n_w = info.treat_dt, info.n_period, info.n_treatment
        lambd = info.lambd
        coef = info.coef
        min_num = self.info.min_point_num_node
        configs = {leaf.level_id: {fid: [0, n_bins] for fid in range(m)} for leaf in leaves}
        parameters = f"""{{
            "tree": {{
                "lambd": {lambd},
                "coeff": {coef},
                "min_point_num_node": {min_num},
                "min_var_rate": {0.1},
                "monotonic_constraints": {self.w_monotonic},
                "parallel_l2": {self.info.parallel_l2}
            }},
            
            "threads": {self.nthreads},
            "dataset": {{
                "treat_dt": {t0}
            }}
        }}"""

        res = didtree_splitting_losses(
            configs,
            hist.bin_grad_hist,
            hist.bin_hess_hist,
            hist.bin_cgrad_hist,
            hist.bin_chess_hist,
            hist.bin_eta_hist,
            hist.bin_counts,
            parameters,
        )
        split_conds = {}
        for leaf in leaves:
            level_id = leaf.level_id
            if level_id not in res:
                leaf.is_leaf = True
                continue
            # opt_feature, opt_bin_idx, opt_loss
            if bool(np.isinf(res[level_id][2])) is False:
                etas = res[level_id][4]
                split_conds[level_id] = {
                    'feature': res[level_id][0],
                    'threshold': res[level_id][1],
                    'loss': res[level_id][2],
                    'theta': (res[level_id][3][0], res[level_id][3][1]),
                    'eta': (np.asarray([etas[0], -etas[0]]), np.asarray([etas[1], -etas[1]])),
                }
            else:
                leaf.is_leaf = True
        return split_conds

    def split(self, leaves: List[GradientCausalTreeNode], hist: Histogram):
        """
        Split the leaf nodes.

        Arguments:
            leaves: The list of leaf nodes.
            hist: The histogram object.

        Returns:
            split_conds: The split conditions.
        """
        return self._split_cpp(leaves, hist)

    def predict(self, x, w=None, key='effect', out=None):
        """
        Predict the treatment effect or other outcomes for given data.

        Arguments:
            x: The input data.
            w: The treatment assignments (default: None).
            key: The key specifying the prediction type (default: 'effect').
            out: The output array (default: None).

        Returns:
            pred: The predicted values.
            cpred: The counterfactual predicted values.
            eta: The optimal parallel interval between the treated and control group.
        """
        if key == 'cf_outcomes':
            pred, cpred, _ = super().predict(x, w, key, out)
            eta = self.predict(x, key='eta')
            temp = np.zeros([cpred.shape[0], 1], dtype=cpred.dtype)
            indexbyarray(eta, w, temp)
            cpred -= temp
            return pred, cpred, eta
        return self._predict(self.export()[0], x, key, out)

    def _predict(self, nodes, x, key='effect', out=None):
        """
        Predicting for the given data using the exported nodes.

        Arguments:
            nodes: The exported nodes.
            x: The input data.
            key: The key specifying the prediction type (default: 'effect').
            out: The output array (default: None).

        Returns:
            The predictions.
        """
        assert isinstance(nodes, list) and len(nodes) > 0, f'nodes must be list and at least one element!'
        if key == 'eta':
            shape = (x.shape[0], self.info.n_treatment)
            x = to_row_major(x)
            if out is None:
                out = np.zeros(shape, np.float64)
            predict(nodes, x, out, key)
            return out
        return super()._predict(nodes, x, key, out)

    def export(self):
        """
        Export the GradientDiDCausalTree.

        Returns:
            slim_cppnodes: The exported nodes in C++ object.
            slim_nodes: The exported nodes in python object.
        """
        nodes, queue = [], [self.root]
        while len(queue) > 0:
            nodes.append(queue.pop(0))
            for child in nodes[-1].children:
                queue.append(child)
        # encode for each node
        slim_cppnodes, slim_nodes = [], []
        for child in nodes:
            effect, debiased_effect = [], []
            for _w in range(1, self.info.n_treatment):
                effect.append(child.theta[_w] - child.theta[0])
                debiased_effect.append(effect[-1] + child.eta[_w])
            info = {
                'leaf_id': child.leaf_id,
                'level_id': child.level_id,
                'outcomes': child.theta,
                'predict': child.theta,
                'eta': child.eta,
                'effect': np.array(effect),
                'debiased_effect': np.array(debiased_effect),
                'is_leaf': child.is_leaf,
                'children': [-1, -1],
                'split_feature': -1,
                'split_thresh': -1,
            }
            if child.is_leaf is False:
                info['children'] = [nodes.index(child.children[0]), nodes.index(child.children[1])]
                info['split_feature'] = child.split_feature
                info['split_thresh'] = child.split_rawthresh
            slim_cppnodes.append(create_didnode_from_dict(info))
            info['split_feature'] = self.info.feature_columns[info['split_feature']]
            slim_nodes.append(info)
        return slim_cppnodes, slim_nodes
