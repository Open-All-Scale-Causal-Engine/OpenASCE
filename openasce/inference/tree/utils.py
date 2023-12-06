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

from typing import List

import numpy as np
import pandas as pd
import logging

from openasce.utils.logger import logger
from .gbct_utils.common import data as data_utils
from .gbct_utils import common, bin

indexbyarray_fn_map = {
    "double_int32": common.indexbyarray2_double_int32,
    "double_uint32": common.indexbyarray2_double_uint32,
    "float64_int32": common.indexbyarray2_double_int32,
    "float64_uint32": common.indexbyarray2_double_uint32,
    "float32_int32": common.indexbyarray2_float_int32,
    "float32_uint32": common.indexbyarray2_float_uint32,
    "float32_int64": common.indexbyarray2_float_int64,
    "float32_uint64": common.indexbyarray2_float_uint64,
}

update_histogram_fn_map = {
    "update_histogram_int32_int32": common.update_histogram_int32_int32,
    "update_histogram_double_int32": common.update_histogram_double_int32,
    "update_histogram_float64_int32": common.update_histogram_double_int32,
    "update_histogram_float32_int32": common.update_histogram_float32_int32,
    "update_histograms_int32_int32": common.update_histograms_int32_int32,
    "update_histograms_double_int32": common.update_histograms_double_int32,
    "update_histograms_float64_int32": common.update_histograms_double_int32,
    "update_histograms_float32_int32": common.update_histograms_float32_int32,
}

Value2BinParallel_fn_map = {
    "double_int32": bin.Value2BinParallel_double_int32,
    "double_uint32": bin.Value2BinParallel_double_uint32,
    "float64_int32": bin.Value2BinParallel_double_int32,
    "float64_uint32": bin.Value2BinParallel_double_uint32,
    "float32_int32": bin.Value2BinParallel_float_int32,
    "float32_uint32": bin.Value2BinParallel_float_uint32,
}

groupby_fn_map = {
    "float64_int32": data_utils.groupby_double_int32,
    "float64_uint32": data_utils.groupby_double_uint32,
    "double_int32": data_utils.groupby_double_int32,
    "double_uint32": data_utils.groupby_double_uint32,
    "float64_int64": data_utils.groupby_double_int64,
    "float64_uint64": data_utils.groupby_double_uint64,
    "double_int64": data_utils.groupby_double_int64,
    "double_uint64": data_utils.groupby_double_uint64,
    "float32_int64": data_utils.groupby_float_int64,
    "float32_uint64": data_utils.groupby_float_uint64,
    "float32_int32": data_utils.groupby_float_int32,
    "float32_uint32": data_utils.groupby_float_uint32,
}


def t_or_f(arg):
    ua = str(arg).upper()
    if "TRUE".startswith(ua):
        return True
    elif "FALSE".startswith(ua):
        return False
    else:
        raise ValueError(f"unknown parameter!")


def list_to_array(data: list, out=None, st_idx: int = 0, miss_value=0, threads=-1):
    n = len(data)
    m = len(data[0])
    odtype = type(data[0][0])
    if out is None:
        out = np.full([n, m], miss_value, dtype=odtype)
    fn_name = f"fill_array_fast_{out.dtype.name}"
    fn = getattr(data_utils, fn_name)
    fn(data, out, st_idx, miss_value, threads)
    return out


def groupby(data: np.ndarray, by: np.ndarray, aggregator: str = "mean", dropna=True):
    dtype = data.dtype
    itype = by.dtype
    fn_key = f"{dtype.name}_{itype.name}"
    fn = groupby_fn_map[fn_key]
    return fn(data.astype(float), by.astype(np.int32), aggregator, dropna)


def to_row_major(x, dtype=None):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy() if dtype is None else x.to_numpy(dtype)
    if x.flags.c_contiguous is False:
        x = np.ascontiguousarray(x) if dtype is None else np.ascontiguousarray(x, dtype)
    return x


trace_level = 15
trace_name = "TRACE"


def set_log_level_cpp(level):
    level_map = {
        logging.DEBUG: common.LogLevel.Debug,
        logging.INFO: common.LogLevel.Info,
        15: common.LogLevel.Trace,
        logging.WARNING: common.LogLevel.Warning,
        logging.FATAL: common.LogLevel.Fatal,
    }
    common.reset_loglevel(level_map[level])


def init_logger():
    logging.addLevelName(trace_level, trace_name)
    set_log_level_cpp(logger.level)


init_logger()


def DEBUG(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)


def TRACE(msg, *args, **kwargs):
    logger.log(trace_level, msg, *args, **kwargs)


def INFO(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)


def WARN(msg, *args, **kwargs):
    logger.warn(msg, *args, **kwargs)


def ERROR(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)


def FATAL(msg, *args, **kwargs):
    logger.fatal(msg, *args, **kwargs)


def _check_match(*args, axis=0):
    n = None
    for arg in args:
        assert len(arg.shape) > axis, f"The given axis({axis}) out of range!"
        if n is None:
            n = arg.shape[axis]
        assert (
            n == arg.shape[axis]
        ), f"The shape on dimension of axis={axis} doesn't match!"


def _check_c_style_array(*args):
    for array in args:
        if isinstance(array, np.ndarray):
            assert array.flags.c_contiguous, f"input array must be c_style stored!"


def indexbyarray(arr, idx, fact_outcome, counterfact_outcome=None, n_threads=-1):
    """
    Index an outcome array (arr with shape [n, 2]) by another binary treament array (idx)

    Arguments:
        arr (ndarray): The input array.
        idx (ndarray): The index array.
        fact_outcome (ndarray): The outcome array to update.
        counterfact_outcome (ndarray): The counterfactual outcome array to update.
        n_threads (int, optional): The number of threads to use. Defaults to -1.

    Returns:
        ndarray: The updated outcome arrays.

    """
    assert (
        arr.dtype == fact_outcome.dtype
    ), f"arr.dtype({arr.dtype}) != fact_outcome.dtype({fact_outcome.dtype})!"
    assert (
        arr.shape[0] == idx.shape[0]
    ), f"arr.shape({arr.shape}) != idx.shape({idx.shape})!"
    assert (
        arr.shape[0] == fact_outcome.shape[0]
    ), f"arr.shape({arr.shape}) != out.shape({fact_outcome.shape})!"
    if counterfact_outcome is not None:
        assert (
            fact_outcome.shape == counterfact_outcome.shape
            and counterfact_outcome.dtype == fact_outcome.dtype
        ), f"fact_outcome and counterfact_outcome should be the same shape and dtype!"
    else:
        counterfact_outcome = np.array([], fact_outcome.dtype)
    fn = indexbyarray_fn_map[f"{arr.dtype.name}_{idx.dtype.name}"]
    return fn(arr, idx, fact_outcome, counterfact_outcome, n_threads)


def update_x_map(x_binned, ins2leaf, split_infos, leaves_range, out, nthread=-1):
    """
    Update the index of instances

    Arguments:
        x_binned (ndarray): The binned feature array.
        ins2leaf (ndarray): The mapping array.
        split_infos (ndarray): The split information array.
        leaves_range (ndarray): The range of each leaf.
        out (ndarray): The output array to store the updated mapping.
        nthread (int, optional): The number of threads to use. Defaults to -1.

    Returns:
        None

    """
    assert (
        x_binned.shape[0] == ins2leaf.shape[0]
        and split_infos.shape[0] == leaves_range.shape[0]
    )
    dtype = x_binned.dtype
    assert (
        dtype == ins2leaf.dtype
        and dtype == split_infos.dtype
        and dtype == leaves_range.dtype
        and dtype == out.dtype
    )
    _check_c_style_array(x_binned, ins2leaf, out)
    common.update_x_map_int32(
        x_binned, ins2leaf, split_infos, leaves_range, out, nthread
    )


def update_histogram(
    target,
    x_binned,
    index,
    leaves_range,
    treatment,
    out,
    leaves=[],
    n_treatment=2,
    n_bins=64,
    threads=-1,
):
    """
    Update the histogram of each leaf.

    Arguments:
        target (ndarray): Target array. Shape [n, n_outcome].
        x_binned (ndarray): Binned feature array. Shape [n, n_feature].
        index (ndarray): Index array. Shape [n]. The end position in leaves_range must not exceed n.
        leaves_range (ndarray): List of each leaf's data range. Shape [n_leaf, 2]. Each term looks like [st_pos, end_pos).
        treatment (ndarray): Treatment array. Shape [n].
        out (ndarray): Output histogram. Shape [n_leaf, n_features, n_bins, n_treatment, n_outcome].
        leaves (list, optional): List of leaf indices. Defaults to [].
        n_treatment (int, optional): The number of treatments. Defaults to 2.
        n_bins (int, optional): The number of bins. Defaults to 64.
        threads (int, optional): The number of threads to use. Defaults to -1.

    Returns:
        ndarray: The updated histogram array.

    """
    dtype = target.dtype
    itype = x_binned.dtype
    assert (
        dtype == out.dtype
    ), f"the `target`({dtype}) must be the same with `out`({out.dtype})"
    assert (
        itype == x_binned.dtype
        and itype == index.dtype
        and itype == leaves_range.dtype
        and itype == treatment.dtype
    ), (
        f"expect `x_binned`({x_binned.dtype}), `leaves_range`({leaves_range.dtype}),"
        "`treatment`({treatment.dtype}) and `index`({index.dtype}) be the same dtype!"
    )
    n_outcome = target.shape[1]
    n_feature = x_binned.shape[1]
    n_leaf = len(leaves_range)
    n_i = index.shape[0]
    _check_match(target, x_binned, treatment)
    _check_c_style_array(target, x_binned, index, leaves_range, treatment, out)
    for _, end in leaves_range:
        assert end <= n_i, f"leaf range out of `index`"
    # out shape check
    assert out.shape == (n_leaf, n_feature, n_bins, n_treatment, n_outcome), (
        f"the shape of `out` ({out.shape}) not "
        f"equals to {[n_leaf, n_feature, n_bins, n_treatment, n_outcome]}!"
    )
    fn_key = f"update_histogram_{dtype.name}_{itype.name}"
    fn = update_histogram_fn_map[fn_key]
    return fn(
        target,
        x_binned,
        index,
        leaves_range,
        treatment,
        out,
        leaves,
        n_treatment,
        n_bins,
        threads,
    )


def update_histograms(
    targets,
    x_binned,
    index,
    leaves_range,
    treatment,
    outs,
    leaves=[],
    n_treatment=2,
    n_bins=64,
    threads=-1,
):
    """
    Update the histogram of each leaf.

    Arguments:
        targets (list): List of target arrays. Shape [n, n_outcome].
        x_binned (ndarray): Binned feature array. Shape [n, n_feature].
        index (ndarray): Index array. Shape [n]. Must satisfy that the end position in leaves_range is not greater than n.
        leaves_range (ndarray): List of each leaf's data range. Shape [n_leaf, 2]. Each term looks like [st_pos, end_pos).
        treatment (ndarray): Treatment array. Shape [n].
        outs (list): List of output histogram arrays. Shape [n_leaf, n_features, n_bins, n_treatment, n_outcome].
        leaves (list, optional): List of leaf indices. Defaults to [].
        n_treatment (int, optional): The number of treatments. Defaults to 2.
        n_bins (int, optional): The number of bins. Defaults to 64.
        threads (int, optional): The number of threads to use. Defaults to -1.

    Returns:
        ndarray: The updated histogram arrays.

    """
    dtype = targets[0].dtype
    assert np.all(
        [(dtype == t.dtype) for t in targets]
    ), "The dtype of elements in targets must be the same!"
    itype = x_binned.dtype
    for out in outs:
        assert (
            dtype == out.dtype
        ), f"the `target`({dtype}) must be the same with `out`({out.dtype})"
    assert (
        itype == x_binned.dtype
        and itype == index.dtype
        and itype == leaves_range.dtype
        and itype == treatment.dtype
    ), (
        f"expect `x_binned`({x_binned.dtype}), `leaves_range`({leaves_range.dtype}),"
        f"`treatment`({treatment.dtype}) and `index`({index.dtype}) be the same dtype!"
    )
    n_feature = x_binned.shape[1]
    n_leaf = len(leaves_range)
    n_i = index.shape[0]
    [_check_match(target, x_binned, treatment) for target in targets]
    [
        _check_c_style_array(target, x_binned, index, leaves_range, treatment, out)
        for target, out in zip(targets, outs)
    ]
    for _, end in leaves_range:
        assert end <= n_i, f"leaf range out of `index`"
    # out shape check
    for target, out in zip(targets, outs):
        _shape = (n_leaf, n_feature, n_bins, n_treatment, target.shape[1])
        assert (
            out.shape == _shape
        ), f"the shape of `out` ({out.shape}) not equals to {_shape}!"
    fn_key = f"update_histograms_{dtype.name}_{itype.name}"
    fn = update_histogram_fn_map[fn_key]
    return fn(
        targets,
        x_binned,
        index,
        leaves_range,
        treatment,
        outs,
        leaves,
        n_treatment,
        n_bins,
        threads,
    )


def find_bin_parallel(
    data,
    max_bin=64,
    min_data_in_bin=100,
    min_split_data=100,
    pre_filter=False,
    bin_type=0,
    use_missing=True,
    zero_as_missing=False,
    forced_upper_bounds=[],
):
    """
    Find bins in parallel for the given data.

    Arguments:
        data: The input data.
        max_bin: The maximum number of bins. (default: 64)
        min_data_in_bin: The minimum number of data points in a bin. (default: 100)
        min_split_data: The minimum number of data points to split a bin. (default: 100)
        pre_filter: Whether to pre-filter the data. (default: False)
        bin_type: The type of binning. (default: 0)
        use_missing: Whether to use missing values. (default: True)
        zero_as_missing: Whether to treat zero as a missing value. (default: False)
        forced_upper_bounds: The forced upper bounds for the bins. (default: [])

    Returns:
        The bins found.

    Raises:
        ValueError: If the data type is not supported.

    """
    dtype = data.dtype
    _check_c_style_array(data)
    if dtype.name in ("double", "float64"):
        return bin.FindBinParallel_double(
            data,
            max_bin,
            min_data_in_bin,
            min_split_data,
            pre_filter,
            bin_type,
            use_missing,
            zero_as_missing,
            forced_upper_bounds,
        )
    else:
        raise ValueError(f"The {dtype.name} has not been supported!")


def value_bin_parallel(data, bin_mappers: List[bin.BinMaper], out=None, threads=-1):
    """
    Transform the input data to bin values in parallel.

    Arguments:
        data: The input data.
        bin_mappers: The bin mappers.
        out: The output array to store the bin values. (default: None)
        threads: The number of threads to use (-1 for maximum). (default: -1)

    Returns:
        The transformed bin values.

    Raises:
        ValueError: If the output dtype is not supported.

    """
    if out is None:
        out = np.zeros_like(data, np.int32)
    assert out.dtype in (
        np.int32,
        np.uint32,
    ), f"out dtype must be {np.int32} or {np.uint32}!"
    _check_c_style_array(data, out)
    dtype = data.dtype
    idtype = out.dtype
    fn_key = f"{dtype.name}_{idtype.name}"
    fn = Value2BinParallel_fn_map[fn_key]

    assert isinstance(data, np.ndarray), f"Only np.ndarray is supported!"
    return fn(data, bin_mappers, out, threads)
