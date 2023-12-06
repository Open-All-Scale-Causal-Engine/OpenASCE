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

#ifndef __DIFFERENCE_IN_DIFFERENCES_LOSSES_H__
#define __DIFFERENCE_IN_DIFFERENCES_LOSSES_H__
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
#include "gbct_loss.h"

namespace py = pybind11;

namespace gbct_utils {

template <typename dtype, typename idtype>
std::tuple<dtype, std::unique_ptr<dtype[]>, dtype>
_didtree_loss(const std::unique_ptr<dtype[]> &pgrad,    // shape: n_treatment, n_outs
              const std::unique_ptr<dtype[]> &phess,    // shape: n_treatment, n_outs
              const std::unique_ptr<dtype[]> &pcgrad,   // shape: n_treatment, n_outs
              const std::unique_ptr<dtype[]> &pchess,   // shape: n_treatment, n_outs
              const std::unique_ptr<dtype[]> &pre_eta,  // shape: n_treatment
              const std::unique_ptr<idtype[]> &pcounts, // shape: n_treatment
              size_t n_treatments, size_t n_outs, dtype lambd, dtype parallel_l2, dtype coeff, int t0,
              int monotonic_constraints) {
    // size_t n_treatments = grad.shape(0), n_outs = grad.shape(1);
    dtype loss = 0;
    size_t total_counts = sum(pcounts, {n_treatments});
    std::unique_ptr<dtype[]> ptotal_g(new dtype[n_outs]()), ptotal_h(new dtype[n_outs]()),
        pcg(new dtype[n_outs * n_treatments]()), pch(new dtype[n_outs * n_treatments]());
    // step 1, estimate parameters
    auto pyhat = estimate_by_grad(pgrad, phess, {n_treatments, n_outs}, lambd);
    for (size_t i = 0; i < n_treatments; i++) {
        for (size_t j = 0; j < n_outs; j++) {
            auto offset = i * n_outs + j;
            ptotal_g[j] += pgrad[offset];
            ptotal_h[j] += phess[offset];
        }
    }
    for (size_t i = 0; i < n_treatments; i++) {
        for (size_t j = 0; j < n_outs; j++) {
            auto offset = i * n_outs + j;
            pcg[offset] = ptotal_g[j] - pgrad[offset];
            pch[offset] = ptotal_h[j] - phess[offset];
        }
    }
    auto pcyhat = estimate_by_grad(pcg, pch, {n_treatments, n_outs}, lambd);
    // step 2 calculate eta
    dtype numerator = 0, denominator = 0;
    dtype mean_eta = (pre_eta[0] / pcounts[0] + pre_eta[1] / pcounts[1])/2;// only for binary
    for (size_t d = 1; d < n_treatments; d++) {
        for (size_t t = 0; t < t0; t++) { // pre-treatment
            auto idx = d * n_outs + t;
            numerator += (pcgrad[idx] / pcounts[d] - pcgrad[t] / pcounts[0]);
            numerator += (pchess[idx] * pcyhat[idx] / pcounts[d] - pchess[t] * pcyhat[t] / pcounts[d]);
            numerator += parallel_l2 * mean_eta;
            denominator += pchess[idx] / pcounts[d] + pchess[t] / pcounts[0];
        }
    }
    dtype local_eta = -numerator / (denominator + t0 * parallel_l2);
    // step 3, calculate loss
    // (\sum_{i\in treats} \ell(y_i, {hat_y}_i)/{|treats|} + \sum_{i\in controls} l(y_i,
    // {hat_y}_i)/{|controls|})*|treats \cup controls|/2
    for (size_t d = 0; d < n_treatments; d++) {
        dtype numerator = 0, denominator = pcounts[d] * t0;
        for (size_t t = 0; t < t0; t++) { // pre-treatment
            auto idx = d * n_outs + t;
            numerator += loss_by_grad(pcyhat[idx] + (2 * d - 1.0) * local_eta, pcgrad[idx], pchess[idx], lambd);
        }
        if (t0 > 0) { loss += numerator / denominator; }
        numerator = 0;
        denominator = pcounts[d] * (n_outs - t0);
        for (size_t t = t0; t < n_outs; t++) { // post-treatment
            auto idx = d * n_outs + t;
            numerator += loss_by_grad(pyhat[idx], pgrad[idx], phess[idx], lambd);
        }
        loss += coeff * numerator / denominator;
    }
    // monotonic_constraints
    if (monotonic_constraints != 0) {
        for (size_t t = t0; t < n_outs; ++t) {
            auto tau = pyhat[n_outs + t] - pyhat[t];
            if (monotonic_constraints * tau < 0) loss += std::numeric_limits<dtype>::infinity();
        }
    }
    if (parallel_l2 > 0) {
        dtype parallel_l2_reg = (.5 * local_eta * local_eta + local_eta * mean_eta);
        loss += parallel_l2 * parallel_l2_reg;
    }

    return std::make_tuple(loss * total_counts / 2, std::move(pyhat), local_eta);
}

template <typename dtype, typename idtype>
std::tuple<size_t, dtype, std::unique_ptr<dtype[]>, std::unique_ptr<dtype[]>>
_didtree_loss_on_feature(size_t fidx, const std::vector<size_t> &bins_range,
                         const py::array_t<dtype> &grad,    // shape: n_feature, n_bins, n_treatment, n_outs
                         const py::array_t<dtype> &hess,    // shape: n_feature, n_bins, n_treatment, n_outs
                         const py::array_t<dtype> &cgrad,   // shape: n_feature, n_bins, n_treatment, n_outs
                         const py::array_t<dtype> &chess,   // shape: n_feature, n_bins, n_treatment, n_outs
                         const py::array_t<dtype> &eta,     // shape: n_feature, n_bins, n_treatment
                         const py::array_t<idtype> &counts, // shape: n_feature, n_bins, n_treatment
                         dtype lambd, dtype parallel_l2, dtype coeff, dtype min_var_rate, size_t min_points, int t0,
                         int monotonic_constraints) {
    size_t n_feat = grad.shape(0);
    size_t n_bins = grad.shape(1);
    size_t n_treats = grad.shape(2);
    size_t n_outs = grad.shape(3);
    dtype opt_loss = std::numeric_limits<dtype>::infinity();
    size_t opt_bin_idx = std::numeric_limits<size_t>::infinity();
    std::unique_ptr<dtype[]> opt_y(new dtype[n_treats * n_outs * 2]());
    std::unique_ptr<dtype[]> opt_eta(new dtype[2]());
    // create_array
    auto l_grad = std::unique_ptr<dtype[]>(new dtype[n_treats * n_outs]());
    auto l_hess = std::unique_ptr<dtype[]>(new dtype[n_treats * n_outs]());
    auto l_cgrad = std::unique_ptr<dtype[]>(new dtype[n_treats * n_outs]());
    auto l_chess = std::unique_ptr<dtype[]>(new dtype[n_treats * n_outs]());
    auto l_eta = std::unique_ptr<dtype[]>(new dtype[n_treats]());
    auto l_count = std::unique_ptr<idtype[]>(new idtype[n_treats]());
    auto r_grad = std::unique_ptr<dtype[]>(new dtype[n_treats * n_outs]());
    auto r_hess = std::unique_ptr<dtype[]>(new dtype[n_treats * n_outs]());
    auto r_cgrad = std::unique_ptr<dtype[]>(new dtype[n_treats * n_outs]());
    auto r_chess = std::unique_ptr<dtype[]>(new dtype[n_treats * n_outs]());
    auto r_count = std::unique_ptr<idtype[]>(new idtype[n_treats]());
    auto r_eta = std::unique_ptr<dtype[]>(new dtype[n_treats]());

    // variance for treatment
    std::vector<dtype> cum_treat_squared(n_bins, 0);
    std::vector<idtype> cum_count(n_bins, 0);

    for (size_t i = 0; i < n_bins; i++) {
        for (size_t m = 0; m < n_treats; m++) {
            for (size_t n = 0; n < n_outs; n++) {
                auto offset = m * n_outs + n;
                r_grad[offset] += grad.at(fidx, i, m, n);
                r_hess[offset] += hess.at(fidx, i, m, n);
                r_cgrad[offset] += cgrad.at(fidx, i, m, n);
                r_chess[offset] += chess.at(fidx, i, m, n);
            }
            r_count[m] += counts.at(fidx, i, m);
            r_eta[m] += eta.at(fidx, i, m);
        }
        if (min_var_rate > 0) {
            cum_treat_squared[i] = r_count[1];
            cum_count[i] = sum(r_count, {n_treats});
        }
    }

    dtype variance = (cum_treat_squared[n_bins - 1] / cum_count[n_bins - 1]
                      - std::pow(cum_treat_squared[n_bins - 1] / cum_count[n_bins - 1], 2));

    auto cal_cnt = 0;
    for (size_t i = 0; i < bins_range[1]; i++) {
        for (size_t m = 0; m < n_treats; m++) {
            for (size_t n = 0; n < n_outs; n++) {
                auto offset = m * n_outs + n;
                // left
                l_grad[offset] += grad.at(fidx, i, m, n);
                l_hess[offset] += hess.at(fidx, i, m, n);
                l_cgrad[offset] += cgrad.at(fidx, i, m, n);
                l_chess[offset] += chess.at(fidx, i, m, n);

                // right
                r_grad[offset] -= grad.at(fidx, i, m, n);
                r_hess[offset] -= hess.at(fidx, i, m, n);
                r_cgrad[offset] -= cgrad.at(fidx, i, m, n);
                r_chess[offset] -= chess.at(fidx, i, m, n);
            }
            l_eta[m] += eta.at(fidx, i, m);
            l_count[m] += counts.at(fidx, i, m);
            r_eta[m] -= eta.at(fidx, i, m);
            r_count[m] -= counts.at(fidx, i, m);
        }
        if (i >= bins_range[0]) {
            if (min_var_rate > 0) {
                // left treatment variance: variance(y) = E(y^2) - E(y) * E(y)
                dtype left_var =
                    (cum_treat_squared[i] - std::pow(cum_treat_squared[i], 2) / cum_count[i]) / cum_count[i];
                dtype right_squared_treat = std::pow(cum_treat_squared[n_bins - 1] - cum_treat_squared[i], 2);
                auto right_count = (cum_count[n_bins - 1] - cum_count[i]) + 1e-9;
                dtype right_var = (right_squared_treat - right_squared_treat / right_count) / right_count;
                if (std::min(right_var, left_var) < min_var_rate * variance) {
                    TRACE("The min of children's variance(%.2f) is too small compared with parent(%.2f), "
                          "therefore skip current splitting on (feature: %d@bin: %d)!",
                          std::min(right_var, left_var), variance, fidx, i);
                    continue;
                }
            }
            if (min_points > 0) {
                auto _min_points = std::min(*std::min_element(l_count.get(), l_count.get() + n_treats),
                                            *std::min_element(r_count.get(), r_count.get() + n_treats));
                if (_min_points < min_points) {
                    TRACE("The min of children's points is too small (%d<%d), therefore skip current splitting on ("
                          "feature: %d@bin: %d)!",
                          _min_points, min_points, fidx, i);
                    continue;
                }
            }

            auto left = _didtree_loss(l_grad, l_hess, l_cgrad, l_chess, l_eta, l_count, n_treats, n_outs, lambd,
                                      parallel_l2, coeff, t0, monotonic_constraints);
            auto right = _didtree_loss(r_grad, r_hess, r_cgrad, r_chess, r_eta, r_count, n_treats, n_outs, lambd,
                                       parallel_l2, coeff, t0, monotonic_constraints);

            if (std::get<0>(left) + std::get<0>(right) < opt_loss) {
                opt_loss = std::get<0>(left) + std::get<0>(right);
                opt_bin_idx = i;
                size_t size = n_treats * n_outs * sizeof(dtype);
                memcpy(opt_y.get(), std::get<1>(left).get(), size);
                memcpy(opt_y.get() + n_treats * n_outs, std::get<1>(right).get(), size);
                opt_eta[0] = std::get<2>(left);
                opt_eta[1] = std::get<2>(right);
            }
            cal_cnt += 1;
        }
    }
    if (opt_loss == std::numeric_limits<dtype>::infinity()) {
        TRACE("feature: %d has been calulated %d!", fidx, cal_cnt);
    }

    return std::make_tuple(opt_bin_idx, opt_loss, std::move(opt_y), std::move(opt_eta));
}

template <typename dtype, typename idtype>
std::tuple<size_t, size_t, dtype, std::unique_ptr<dtype[]>, std::unique_ptr<dtype[]>>
_didtree_loss_on_node(const std::unordered_map<size_t, std::vector<size_t>> &features, //{features: [bin_from, bin_to]}
                      const py::array_t<dtype> &grad,    // shape: n_feature, n_bins, n_treatment, n_outs
                      const py::array_t<dtype> &hess,    // shape: n_feature, n_bins, n_treatment, n_outs
                      const py::array_t<dtype> &cgrad,   // shape: n_feature, n_bins, n_treatment, n_outs
                      const py::array_t<dtype> &chess,   // shape: n_feature, n_bins, n_treatment, n_outs
                      const py::array_t<dtype> &eta,     // shape: n_feature, n_bins, n_treatment, 1
                      const py::array_t<idtype> &counts, // shape: n_feature, n_bins, n_treatment
                      dtype lambd, dtype parallel_l2, dtype coeff, dtype min_var_rate, size_t min_points, int t0,
                      int monotonic_constraints, int threads) {
    size_t n_feature = grad.shape(0);
    size_t n_bins = grad.shape(1);
    size_t n_treatments = grad.shape(2);
    size_t n_outs = grad.shape(3);
    dtype opt_loss = std::numeric_limits<dtype>::infinity();
    size_t opt_bin_idx = std::numeric_limits<size_t>::infinity();
    size_t opt_feature = std::numeric_limits<size_t>::infinity();
    std::unique_ptr<dtype[]> opt_y;
    std::unique_ptr<dtype[]> opt_eta;
    std::unordered_map<size_t,
                       std::future<std::tuple<size_t, dtype, std::unique_ptr<dtype[]>, std::unique_ptr<dtype[]>>>>
        losses;

    py::gil_scoped_release release;
    // create thread pool
    ThreadPool pool(threads);
    for (auto item : features) {
        auto fid = item.first;
        auto &bin_range = item.second;
        losses[fid] =
            pool.enqueue(_didtree_loss_on_feature<dtype, idtype>, fid, bin_range, grad, hess, cgrad, chess, eta, counts,
                         lambd, parallel_l2, coeff, min_var_rate, min_points, t0, monotonic_constraints);
    }
    for (auto &&item : losses) {
        auto i = item.first;
        auto reults = item.second.get();
        if (std::get<1>(reults) < opt_loss) {
            opt_loss = std::get<1>(reults);
            opt_feature = i;
            opt_bin_idx = std::get<0>(reults);
            opt_y = std::move(std::get<2>(reults));
            opt_eta = std::move(std::get<3>(reults));
        }
        DEBUG("feature=%d, bin=%d, loss=%.3f", i, std::get<0>(reults), std::get<1>(reults));
    }

    return std::make_tuple(opt_feature, opt_bin_idx, opt_loss, std::move(opt_y), std::move(opt_eta));
}

template <typename dtype, typename idtype>
std::unordered_map<size_t, std::tuple<size_t, size_t, dtype, py::array_t<dtype>, py::array_t<dtype>>>
didtree_splitting_loss(std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>>
                           candidates,                    // {leaves: {features: bins}}
                       const py::array_t<dtype> &grad,    // shape: n_leaves, n_feature, n_bins, n_treatment, n_outs
                       const py::array_t<dtype> &hess,    // shape: n_leaves, n_feature, n_bins, n_treatment, n_outs
                       const py::array_t<dtype> &cgrad,   // shape: n_leaves, n_feature, n_bins, n_treatment, n_outs
                       const py::array_t<dtype> &chess,   // shape: n_leaves, n_feature, n_bins, n_treatment, n_outs
                       const py::array_t<dtype> &eta,     // shape:  n_leaves, n_feature, n_bins, n_treatment, 1
                       const py::array_t<idtype> &counts, // shape: n_leaves, n_feature, n_bins, n_treatment
                       // dtype lambd, dtype coeff, int t0, int threads,
                       const json11::Json &config) {
    size_t n_leaves = grad.shape(0);
    size_t n_features = grad.shape(1);
    size_t n_bins = grad.shape(2);
    size_t n_treatments = grad.shape(3);
    size_t n_outs = grad.shape(4);

    // parse configure parameters
    auto lambd = config["tree"]["lambd"].number_value();
    auto coeff = config["tree"]["coeff"].number_value();
    auto threads = config["threads"].int_value();
    auto min_point_num_node = config["tree"]["min_point_num_node"].int_value();
    auto min_var_rate = config["tree"]["min_var_rate"].number_value();
    auto t0 = config["dataset"]["treat_dt"].int_value();
    auto monotonic_constraints = config["tree"]["monotonic_constraints"].int_value();
    auto parallel_l2 = config["tree"]["parallel_l2"].number_value();

    DEBUG("lambd:%.2f", lambd);
    DEBUG("coeff:%.2f", coeff);
    DEBUG("threads:%d", threads);
    DEBUG("min_point_num_node:%d", min_point_num_node);
    DEBUG("min_var_rate:%.2f", min_var_rate);
    DEBUG("treated time:%d", t0);
    DEBUG("monotonic_constraints:%d", monotonic_constraints);
    DEBUG("parallel_l2", parallel_l2);

    std::unordered_map<size_t, std::tuple<size_t, size_t, dtype, py::array_t<dtype>, py::array_t<dtype>>> ret;

    for (auto items : candidates) {
        auto i = items.first;
        auto &features = items.second;
        // create_array
        auto _grad = create_array_from_address({n_features, n_bins, n_treatments, n_outs}, grad.data(i));
        auto _hess = create_array_from_address({n_features, n_bins, n_treatments, n_outs}, hess.data(i));
        auto _cgrad = create_array_from_address({n_features, n_bins, n_treatments, n_outs}, cgrad.data(i));
        auto _chess = create_array_from_address({n_features, n_bins, n_treatments, n_outs}, chess.data(i));
        auto _count = create_array_from_address({n_features, n_bins, n_treatments}, counts.data(i));
        auto _eta = create_array_from_address({n_features, n_bins, n_treatments}, eta.data(i));

        auto res = _didtree_loss_on_node(features, _grad, _hess, _cgrad, _chess, _eta, _count, lambd, parallel_l2,
                                         coeff, min_var_rate, min_point_num_node, t0, monotonic_constraints, threads);
        auto ptr = std::get<3>(res).release(), ptr_eta = std::get<4>(res).release();
        ret[i] = std::make_tuple(std::get<0>(res), std::get<1>(res), std::get<2>(res),
                                 std::move(py::array_t<dtype>(std::vector<size_t>({2, n_treatments, n_outs}), ptr)),
                                 std::move(py::array_t<dtype>(std::vector<size_t>({2}), ptr_eta)));
        TRACE("*optimal leaves=%d, feature=%d, bin=%d, loss=%.3f*", i, std::get<0>(res), std::get<1>(res),
              std::get<2>(res));
    }
    if (candidates.size() == 0) { WARNING("You feed an empty split candidates~"); }

    return std::move(ret);
}

} // namespace gbct_utils

#endif // __DIFFERENCE_IN_DIFFERENCES_LOSSES_H__