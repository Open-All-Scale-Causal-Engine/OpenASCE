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

#ifndef __GBCT_PREDICT_H__
#define __GBCT_PREDICT_H__
#include <functional>
#include <unordered_map>
#include <set>

#include "include/utils/common.h"
#include "include/did_utils.h"
#include "include/utils/log.h"

namespace gbct_utils {

// template<typename T, typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
template <typename T>
int predict_single(const std::vector<T> &tree, const py::array_t<double> &x) {
    int rindex = 0;
    while (tree[rindex].is_leaf == false) {
        if (x.at(tree[rindex].split_feature) <= tree[rindex].split_thresh) {
            rindex = tree[rindex].children[0];
        } else {
            rindex = tree[rindex].children[1];
        }
    }
    return rindex;
}

// template<typename T, typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
template <typename T>
void predict_single(const std::vector<std::vector<T>> &trees, py::array_t<double> &out, const py::array_t<double> &x,
                    const std::string &key) {
    if (c_style(out) == false) { throw std::runtime_error("fortune style `out` is not support!"); }
    auto ntree = trees.size();
    auto num_threads = 2 * omp_get_num_procs() - 1;
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < ntree; ++i) {
        auto r = predict_single(trees[i], x);
        array_copy(out.mutable_data(i), trees[i][r].get_info(key));
    }
}

// template<typename T, typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
template <typename T>
void predict_batch(const std::vector<std::vector<T>> &trees,
                   py::array_t<double> &out,     // [n, n_tree, *, n_y]
                   const py::array_t<double> &x, // [n, n_f]
                   const std::string &key, int num_threads) {
    std::unordered_map<std::string, std::function<int(const Node *)>> int_keys = {
        {"leaf_id", &Node::get_leaf_id},
        {"level_id", &Node::get_level_id},
        {"split_feature", &Node::get_split_feature},
        {"is_leaf", &Node::leaf}};

    if (c_style(out) == false) { throw std::runtime_error("fortune style `out` is not support!"); }
    if (c_style(x) == false) { throw std::runtime_error("fortune style `x` is not support!"); }
    TRACE("%d trees, feature shape: (%d, %d), out shape: (%d,%d) and prediction key %s", trees.size(), x.shape(0),
          x.shape(1), out.shape(0), out.shape(1), key.c_str());
    int n = x.shape(0);
    int n_tree = trees.size();
    num_threads = num_threads > 0 ? num_threads : 2 * omp_get_num_procs() - 1;
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n_tree; j++) {
            auto &nodes = trees[j];
            auto rindex = 0;
            while (nodes[rindex].is_leaf == false) {
                if (x.at(i, nodes[rindex].split_feature) <= nodes[rindex].split_thresh) {
                    rindex = nodes[rindex].children[0];
                } else {
                    rindex = nodes[rindex].children[1];
                }
            }
            // get outputs
            if (int_keys.find(key) != int_keys.end()) {
                mutable_at_(out, i, j) = int_keys[key](&trees[j][rindex]);
            } else {
                array_copy(mutable_data_(out, i, j), trees[j][rindex].get_info(key));
            }
        }
    }
}

// template<typename T, typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
template <typename T>
void predict(const std::vector<std::vector<T>> &trees,
             py::array_t<double> &out,     // [n, n_tree, n_y]
             const py::array_t<double> &x, // [n, n_f]
             const std::string &key, int num_threads) {
    if (x.ndim() == 2) {
        predict_batch(trees, out, x, key, num_threads);
    } else if (x.ndim() == 1) {
        predict_single(trees, out, x, key);
    } else {
        LightGBM::Log::Fatal("dimension(x) = %d is not supported!", x.ndim());
    }
}

} // namespace gbct_utils
#endif // __GBCT_PREDICT_H__