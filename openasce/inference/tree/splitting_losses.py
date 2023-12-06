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

import numpy as np

from .gbct_utils import splitting, common


def causal_tree_splitting_losses(configs, bin_outcome_hist, bin_counts, parameters: dict):
    """
    Calculate the splitting losses for the ordinary causal tree.

    Arguments:
        configs: Configuration.
        bin_outcome_hist: Histogram of outcome values.
        bin_counts: Histogram counts.
        parameters: Additional parameters.

    Returns:
        The splitting losses.

    """
    dtype = bin_outcome_hist.dtype
    idtype = bin_counts.dtype
    fn_key = f'causal_tree_splitting_loss_{dtype.name}_{idtype.name}'
    assert hasattr(splitting, fn_key), f'bin_outcome_hist({dtype.name}) and bin_counts({idtype.name}) is not supported!'
    fn = getattr(splitting, fn_key)
    return fn(configs, bin_outcome_hist, bin_counts, common.data.json_from_str(parameters))


def causal_tree_splitting_losses2(configs, bin_grad_hist, bin_hess_hist, bin_counts, parameters: dict):
    """
    Calculate the splitting losses for the ordinary causal tree.

    Arguments:
        configs: Configuration.
        bin_outcome_hist: Histogram of outcome values.
        bin_counts: Histogram counts.
        parameters: Additional parameters.

    Returns:
        The splitting losses.

    """
    dtype = bin_grad_hist.dtype
    idtype = bin_counts.dtype
    assert dtype == bin_hess_hist.dtype, f'the dtype of bin_grad_hist and bin_hess_hist should be the same!'
    fn_key = f'causal_tree_splitting_loss2_{dtype.name}_{idtype.name}'
    assert hasattr(splitting, fn_key), f'bin_grad_hist({dtype.name}) and bin_counts({idtype.name}) is not supported!'
    fn = getattr(splitting, fn_key)
    return fn(configs, bin_grad_hist, bin_hess_hist, bin_counts, common.data.json_from_str(parameters))


def gbct_splitting_losses(
    configs, bin_grad_hist, bin_hess_hist, bin_cgrad_hist, bin_chess_hist, bin_counts, parameters: dict
):
    """
    Calculate the splitting losses for the GBCT model.

    Arguments:
        configs: Configuration.
        bin_grad_hist: Histogram of gradients.
        bin_hess_hist: Histogram of Hessians.
        bin_cgrad_hist: Histogram of cumulative gradients.
        bin_chess_hist: Histogram of cumulative Hessians.
        bin_counts: Histogram counts.
        parameters: Additional parameters.

    Returns:
        The splitting losses.

    """
    dtype = bin_grad_hist.dtype
    idtype = bin_counts.dtype

    assert (
        dtype == bin_hess_hist.dtype and dtype == bin_cgrad_hist.dtype and dtype == bin_chess_hist.dtype
    ), f'expect `bin_hess_hist`({bin_hess_hist.dtype}), `bin_cgrad_hist`({bin_cgrad_hist.dtype})  \
        `bin_chess_hist`({bin_chess_hist.dtype}) be the same dtype!'

    fn_key = f'gbct_splitting_loss_{dtype.name}_{idtype.name}'
    assert hasattr(splitting, fn_key), f'{dtype.name}) and {idtype.name} is not supported!'
    fn = getattr(splitting, fn_key)
    return fn(
        configs,
        bin_grad_hist,
        bin_hess_hist,
        bin_cgrad_hist,
        bin_chess_hist,
        bin_counts,
        common.data.json_from_str(parameters),
    )


def didtree_splitting_losses(
    configs, bin_grad_hist, bin_hess_hist, bin_cgrad_hist, bin_chess_hist, bin_eta_hist, bin_counts, parameters: dict
):
    """
    Calculate the splitting losses for the DiD-Tree model.

    Arguments:
        configs: Configuration.
        bin_grad_hist: Histogram of gradients.
        bin_hess_hist: Histogram of Hessians.
        bin_cgrad_hist: Histogram of cumulative gradients.
        bin_chess_hist: Histogram of cumulative Hessians.
        bin_eta_hist: Histogram of etas.
        bin_counts: Histogram counts.
        parameters: Additional parameters.

    Returns:
        The splitting losses.

    """
    dtype = bin_grad_hist.dtype
    idtype = bin_counts.dtype

    assert (
        dtype == bin_hess_hist.dtype and dtype == bin_cgrad_hist.dtype and dtype == bin_chess_hist.dtype
    ), f'expect `bin_hess_hist`({bin_hess_hist.dtype}), `bin_cgrad_hist`({bin_cgrad_hist.dtype})  \
        `bin_chess_hist`({bin_chess_hist.dtype}) and `bin_eta_hist`({bin_eta_hist.dtype}) be the same dtype!'

    fn_key = f'didtree_splitting_loss_{dtype.name}_{idtype.name}'
    assert hasattr(splitting, fn_key), f'{dtype.name}) and {idtype.name} is not supported!'
    fn = getattr(splitting, fn_key)
    return fn(
        configs,
        bin_grad_hist,
        bin_hess_hist,
        bin_cgrad_hist,
        bin_chess_hist,
        bin_eta_hist,
        bin_counts,
        common.data.json_from_str(parameters),
    )
