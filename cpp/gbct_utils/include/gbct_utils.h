/*!
 * Copyright 2023 AntGroup CO., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef __GBCT_UTILS_H__
#define __GBCT_UTILS_H__
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <functional>
#include <tuple>
#include <memory>
#include <vector>

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "utils.h"
#include "utils/log.h"
#include "utils/nparray_ops.h"
#include "utils/thread_pool.hpp"
#include "utils/json11.h"

namespace py = pybind11;
namespace gbct_utils {
/**
 * @brief
 *
 * @tparam idtypeX
 * @param x_binned [n, n_features]. The binned features
 * @param insidx [n]. The index of each instance
 * @param split_infos [L, 2]. The splitting information of `L` tree nodes, like <feature, threshold>.
 * @param scope [L, 2]. The range of each tree node.
 * @param pos_mid [n]
 */
template <typename idtype>
void update_x_map(const py::array_t<idtype> &x_binned, py::array_t<idtype> &insidx,
                  const py::array_t<idtype> &split_infos, const py::array_t<idtype> &scope,
                  py::array_t<idtype> &pos_mid, int threads = -1) {
    // check shape
    int n = x_binned.shape(0);
    int l = split_infos.shape(0);
    shape_match(x_binned, insidx);
    shape_match(split_infos, scope);

    std::vector<idtype> out(insidx.size());
    if (threads < 0) { // default max
        threads = 2 * omp_get_num_procs() - 1;
    }
    py::gil_scoped_release release;
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int i = 0; i < l; i++) {
        auto split_feature = split_infos.at(i, 0);
        auto split_value = split_infos.at(i, 1);
        auto pos_st = scope.at(i, 0);
        auto pos_end = scope.at(i, 1);
        auto left_cur = pos_st, right_cur = pos_end - 1;

        for (idtype j = pos_st; j < pos_end; j++) {
            if (x_binned.at(insidx.at(j), split_feature) <= split_value) {
                out[left_cur++] = insidx.at(j);
            } else {
                out[right_cur--] = insidx.at(j);
            }
        }
        pos_mid.mutable_at(i << 1, 0) = pos_st;
        pos_mid.mutable_at(i << 1, 1) = left_cur;
        pos_mid.mutable_at((i << 1) + 1, 0) = left_cur;
        pos_mid.mutable_at((i << 1) + 1, 1) = pos_end;
        memcpy(insidx.mutable_data() + pos_st, out.data() + pos_st, (pos_end - pos_st) * sizeof(idtype));
    }
}

template <typename dtype, typename idtype>
py::array_t<dtype> &update_histogram(const py::array_t<dtype> &target, const py::array_t<idtype> &x_binned,
                                     const py::array_t<idtype> &index, const py::array_t<idtype> &leaves_range,
                                     const py::array_t<idtype> &treatment, py::array_t<dtype> &out,
                                     const std::vector<size_t> &leaf_ids, int n_treatment = 2, int n_bins = 64,
                                     int threads = -1) {
    auto n = target.shape(0), m = x_binned.shape(1), l = leaves_range.shape(0);
    int n_y = target.shape(1), n_w = n_treatment;
    if (threads < 0) { // default max
        threads = 2 * omp_get_num_procs() - 1;
    }
    std::vector<size_t> leaves(leaf_ids);
    if (leaves.size() == 0) {
        for (size_t i = 0; i < l; i++) { leaves.push_back(i); }
    }
    py::gil_scoped_release release;
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (size_t fid = 0; fid < m; fid++) {
        // for (size_t leaf_id = 0; leaf_id < l; leaf_id++) {
        for (auto leaf_id : leaves) {
            // from leaves_range[leaf_id, 0] to leaves_range[leaf_id, 1]
            auto pos_st = leaves_range.at(leaf_id, 0), pos_end = leaves_range.at(leaf_id, 1);
            for (size_t _idx = pos_st; _idx < pos_end; _idx++) {
                auto id = index.at(_idx);
                auto bin_value = x_binned.at(id, fid), w_id = treatment.at(id);
                for (size_t y_id = 0; y_id < n_y; y_id++) {
                    out.mutable_at(leaf_id, fid, bin_value, w_id, y_id) += target.at(id, y_id);
                }
            }
        }
    }
    return out;
}

template <typename dtype, typename idtype>
std::vector<py::array_t<dtype>> &
update_histograms(const std::vector<py::array_t<dtype>> &targets, const py::array_t<idtype> &x_binned,
                  const py::array_t<idtype> &index, const py::array_t<idtype> &leaves_range,
                  const py::array_t<idtype> &treatment, std::vector<py::array_t<dtype>> &outs,
                  const std::vector<size_t> &leaf_ids, int n_treatment = 2, int n_bins = 64, int threads = -1) {
    auto m = x_binned.shape(1), l = leaves_range.shape(0);
    int n_w = n_treatment;
    if (threads < 0) { // default max
        threads = 2 * omp_get_num_procs() - 1;
    }
    std::vector<size_t> leaves(leaf_ids);
    if (leaves.size() == 0) {
        for (size_t i = 0; i < l; i++) { leaves.push_back(i); }
    }
    if (targets.size() != outs.size()) {
        FATAL("The length of `targets`(%d) and `outs`(%d) must be the same!", targets.size(), outs.size());
    }

    // parallelism
    std::vector<std::pair<size_t, size_t>> _pairs;
    for (size_t fid = 0; fid < m; fid++) {
        for (auto leaf_id : leaves) { _pairs.push_back(std::make_pair(fid, leaf_id)); }
    }
    py::gil_scoped_release release;
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (auto [fid, leaf_id] : _pairs) {
        // from leaves_range[leaf_id, 0] to leaves_range[leaf_id, 1]
        auto pos_st = leaves_range.at(leaf_id, 0), pos_end = leaves_range.at(leaf_id, 1);
        for (size_t _idx = pos_st; _idx < pos_end; _idx++) {
            auto id = index.at(_idx);
            auto bin_value = x_binned.at(id, fid), w_id = treatment.at(id);
            for (size_t i = 0; i < targets.size(); ++i) {
                auto &target = targets[i];
                auto &out = outs[i];
                for (size_t y_id = 0; y_id < target.shape(1); y_id++) {
                    out.mutable_at(leaf_id, fid, bin_value, w_id, y_id) += target.at(id, y_id);
                }
            }
        }
    }
    return outs;
}
} // namespace gbct_utils
#endif // __GBCT_UTILS_H__