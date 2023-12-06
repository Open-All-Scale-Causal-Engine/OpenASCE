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

#ifndef __CAUSAL_TREE_LOSSES2_H__
#define __CAUSAL_TREE_LOSSES2_H__
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

template <typename dtype, typename idtype>
std::tuple<size_t, size_t, dtype, dtype, dtype>
_causal_tree_loss_on_feature2(const py::array_t<dtype> &grad, const py::array_t<dtype> &hess,
                              const py::array_t<idtype> &counts, const std::vector<size_t> &bin_range, size_t leaf_idx,
                              size_t feature_idx, size_t treated_ts, double coeff, dtype variance, double min_var_rate,
                              double imbalance_penalty, int min_point_num_node) {
    size_t n_bins = grad.shape(2);
    size_t n_treatments = grad.shape(3);
    size_t n_outs = grad.shape(4);

    std::vector total_count(n_treatments, idtype(0));
    std::vector left_count(n_treatments, idtype(0)), right_count(n_treatments, idtype(0));
    std::vector left_respond(n_outs * 2, dtype(0)), right_respond(n_outs * 2, dtype(0));

    dtype opt_loss = -std::numeric_limits<dtype>::infinity();
    size_t opt_bin_idx = std::numeric_limits<size_t>::infinity();
    dtype opt_losses[2];
    for (size_t b = 0; b < n_bins; b++) {
        for (size_t d = 0; d < n_treatments; d++) {
            // total_count[d] += counts.at(leaf_idx, feature_idx, b, d);
            right_count[d] += counts.at(leaf_idx, feature_idx, b, d);
            for (size_t t = 0; t < n_outs; t++) {
                right_respond[t] += grad.at(leaf_idx, feature_idx, b, d, t);
                right_respond[t + n_outs] += hess.at(leaf_idx, feature_idx, b, d, t);
            }
        }
    }
    for (size_t j = 0; j < bin_range[1]; ++j) {
        // number of points in left & right child
        for (size_t d = 0; d < n_treatments; ++d) {
            left_count.at(d) += counts.at(leaf_idx, feature_idx, j, d);
            right_count.at(d) -= counts.at(leaf_idx, feature_idx, j, d);
            for (size_t t = 0; t < n_outs; ++t) {
                left_respond.at(t) += grad.at(leaf_idx, feature_idx, j, d, t);
                right_respond.at(t) -= grad.at(leaf_idx, feature_idx, j, d, t);

                left_respond.at(t + n_outs) += hess.at(leaf_idx, feature_idx, j, d, t);
                right_respond.at(t + n_outs) -= hess.at(leaf_idx, feature_idx, j, d, t);
            }
        }
        idtype left_tot_cnt = std::accumulate(left_count.begin(), left_count.end(), 0);
        idtype right_tot_cnt = std::accumulate(right_count.begin(), right_count.end(), 0);
        // check early stop
        // variance for treatment only for binary treatment
        if (min_var_rate > 0) {
            auto left_variance = left_count.at(1) * left_count.at(0) / std::pow(left_tot_cnt, 2);
            auto right_variance = right_count.at(1) * right_count.at(0) / std::pow(right_tot_cnt, 2);
            auto min_variance = min_var_rate * variance;
            if (left_variance < min_variance || right_variance < min_variance) {
                DEBUG(
                    "The variance of children is too small (%d<%d), therefore skip current splitting on (feature: %d@bin: %d)!",
                    std::min(left_variance, right_variance), min_variance, feature_idx, j);
                continue;
            }
        }

        if (min_point_num_node > 0) {
            auto _min = std::min(*std::min_element(left_count.data(), left_count.data() + n_treatments),
                                 *std::min_element(right_count.data(), right_count.data() + n_treatments));
            if (_min < min_point_num_node) {
                DEBUG("The min of children's points is too small (%d<%d), therefore skip current splitting"
                      " on (feature: %d@bin: %d)!",
                      _min, min_point_num_node, feature_idx, j);
                continue;
            }
        }
        if (j < bin_range[0] || j >= bin_range[1]) { continue; }
        // imbalance_penalty * (1.0 / size_left + 1.0 / size_right);
        dtype _imbalance_loss = 0;
        if (imbalance_penalty > 0) { _imbalance_loss = imbalance_penalty * (1.0 / left_tot_cnt + 1.0 / right_tot_cnt); }
        // calculate loss: n_l * n_r/n^2(\theta_l - \theta_r)^2
        // \theta_l = -\frac{\sum_{i\in L} grad_i}{\sum_{i\in L} hess_i}
        // \theta_r = -\frac{\sum_{i\in R} grad_i}{\sum_{i\in R} hess_i}
        dtype _pre_loss = 0, _post_loss = 0;
        for (size_t t = 0; t < n_outs; t++) {
            dtype theta_l = -left_respond.at(t) / left_respond.at(t + n_outs);
            dtype theta_r = -right_respond.at(t) / right_respond.at(t + n_outs);
            dtype weight = 1; // left_tot_cnt * right_tot_cnt / std::pow(left_tot_cnt + right_tot_cnt, 2);
            if (t < treated_ts) {
                _pre_loss += weight * std::pow(theta_l - theta_r, 2);
            } else {
                _post_loss += weight * std::pow(theta_l - theta_r, 2);
            }
        }

        if (_pre_loss < 0) { FATAL("_pre_loss(%f) is negative!!!", _pre_loss); }
        dtype _loss[2] = {-_pre_loss, _post_loss};
        if (_loss[0] + coeff * _loss[1] - _imbalance_loss > opt_loss) {
            opt_bin_idx = j;
            opt_loss = _loss[0] + coeff * _loss[1] - _imbalance_loss;
            opt_losses[0] = _loss[0];
            opt_losses[1] = _loss[1];
        }
    }
    return std::make_tuple(feature_idx, opt_bin_idx, opt_loss, opt_losses[0], opt_losses[1]);
}

template <typename dtype, typename idtype>
std::unordered_map<size_t, std::tuple<size_t, size_t, dtype>>
causal_tree_splitting_loss2(std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>>
                                candidates,                 // {leaves: {features: bins}}
                            const py::array_t<dtype> &grad, // shape: n_leaves, n_feature, n_bins, n_treatment, n_outs
                            const py::array_t<dtype> &hess, // shape: n_leaves, n_feature, n_bins, n_treatment, n_outs
                            const py::array_t<idtype> &counts, // shape: n_leaves, n_feature, n_bins, n_treatment
                            const json11::Json &config) {
    size_t n_leaves = grad.shape(0);
    size_t n_features = grad.shape(1);
    size_t n_bins = grad.shape(2);
    size_t n_treatments = grad.shape(3);
    size_t n_outs = grad.shape(4);
    // INFO("n_leaves:%d, n_features:%d, m_bins:%d, n_treatments:%d, n_outs:%d", n_leaves, n_features, n_bins,
    // n_treatments, n_outs);

    // parse configure parameters
    auto lambd = config["tree"]["lambd"].number_value();
    auto coeff = config["tree"]["coeff"].number_value();
    size_t threads = config["threads"].int_value();
    auto min_point_num_node = config["tree"]["min_point_num_node"].int_value();
    auto min_var_rate = config["tree"]["min_var_rate"].number_value();
    double imbalance_penalty = config["tree"]["imbalance_penalty"].number_value();
    size_t t0 = config["dataset"]["treat_dt"].int_value();
    int monotonic_constraints = config["tree"]["monotonic_constraints"].int_value();

    DEBUG("lambd:%.2f", lambd);
    DEBUG("coeff:%.2f", coeff);
    DEBUG("threads:%d", threads);
    DEBUG("min_point_num_node:%d", min_point_num_node);
    DEBUG("min_var_rate:%.2f", min_var_rate);
    DEBUG("imbalance_penalty:%.2f", imbalance_penalty);
    DEBUG("treated time:%d", t0);
    DEBUG("monotonic_constraints:%d", monotonic_constraints);
    // (sum_{i\in L} _R_i) ^2/|L| + (sum_{i\in R} _R_i) ^2/|R|

    dtype opt_loss = -std::numeric_limits<dtype>::infinity();
    dtype opt_losses[2];
    size_t opt_bin_idx = std::numeric_limits<size_t>::infinity();
    size_t opt_feature = std::numeric_limits<size_t>::infinity();

    std::unordered_map<size_t, std::tuple<size_t, size_t, dtype>> ret;
    if (threads <= 0) { threads = 32; }
    ThreadPool pool(threads);
    py::gil_scoped_release release;
    // feature_idx, bin_idx, loss, loss1, loss2,
    using feat_ret_dtype = std::tuple<size_t, size_t, dtype, dtype, dtype>;
    // {leaf: [optimal loss on each feature]}
    std::unordered_map<size_t, std::vector<std::future<feat_ret_dtype>>> future_losses;
    std::unordered_map<size_t, std::vector<feat_ret_dtype>> losses;

    for (auto &&items : candidates) {
        auto n = items.first; // leaf index
        auto &features = items.second;
        auto &&total_count = sum(counts.data(n, 0), {n_bins, n_treatments}, 0);
        dtype variance = total_count.at(1) * total_count.at(0) / std::pow(sum(total_count), 2);

        future_losses[n] = std::vector<std::future<feat_ret_dtype>>();
        losses[n] = std::vector<feat_ret_dtype>();

        // total variance for treatment
        for (auto &&item : features) {
            auto feature_idx = item.first;

            if (threads > 0) {
                future_losses[n].push_back(pool.enqueue(_causal_tree_loss_on_feature2<dtype, idtype>, grad, hess,
                                                        counts, item.second, n, feature_idx, t0, coeff, variance,
                                                        min_var_rate, imbalance_penalty, min_point_num_node));
            } else {
                losses[n].push_back(_causal_tree_loss_on_feature2(grad, hess, counts, item.second, n, feature_idx, t0,
                                                                  coeff, variance, min_var_rate, imbalance_penalty,
                                                                  min_point_num_node));
            }
        }
    }
    // get future results
    for (auto &leaf_item : future_losses) {
        auto &leaf_id = leaf_item.first;
        std::vector<feat_ret_dtype> temp;
        for (auto &feature_item : leaf_item.second) { temp.push_back(std::move(feature_item.get())); }
        losses[leaf_id] = std::move(temp);
    }
    // gather results
    for (auto &leaf_item : losses) {
        auto leaf_id = leaf_item.first; // leaf index
        opt_loss = -std::numeric_limits<dtype>::infinity();
        opt_bin_idx = std::numeric_limits<size_t>::infinity();
        opt_feature = std::numeric_limits<size_t>::infinity();
        for (auto &&feat_item : leaf_item.second) {
            if (std::get<2>(feat_item) > opt_loss) {
                // feature_idx, bin_idx, loss, loss1, loss2,
                opt_feature = std::get<0>(feat_item);
                opt_bin_idx = std::get<1>(feat_item);
                opt_loss = std::get<2>(feat_item);
                opt_losses[0] = std::get<3>(feat_item);
                opt_losses[1] = std::get<4>(feat_item);
            }
        }
        ret[leaf_id] = std::make_tuple(opt_feature, opt_bin_idx, opt_loss);
        TRACE("*optimal leaves=%d, feature=%d, bin=%d, loss=%.3f(%.3f, %.3f)*", leaf_id, opt_feature, opt_bin_idx,
              opt_loss, opt_losses[0], opt_losses[1]);
    }

    if (candidates.size() == 0) { TRACE("You feed an empty split candidates~"); }
    return ret;
}

} // namespace gbct_utils

#endif // __CAUSAL_TREE_LOSSES2_H__