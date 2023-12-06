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

#ifndef __DATA_UTILS_H__
#define __DATA_UTILS_H__
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "include/utils/common.h"
#include "include/utils/json11.h"

#include "include/gbct_utils.h"

namespace py = pybind11;

namespace gbct_utils {

std::map<std::string, double> str_map(const std::string &str, char sep1, char sep2) {
    std::map<std::string, double> ret;
    auto temp = LightGBM::Common::Split(str.c_str(), sep1);
    for (size_t i = 0; i < temp.size(); i++) {
        auto pair = LightGBM::Common::Split(temp[i].c_str(), sep2);
        LightGBM::Common::Atof(pair[1].c_str(), &ret[pair[0]]);
    }
    return ret;
}

void parse_kv_str(const std::vector<std::string> &lines, py::array_t<double> out,
                  const std::vector<std::string> &headers, double default_value = 0, char sep1 = ',', char sep2 = ':',
                  int num_threads = -1) {
    auto m = headers.size(), n = lines.size();
    std::vector<int> empty_cnts(n);
    std::set<std::string> head_map(headers.begin(), headers.end());
    if (out.shape(0) != n or out.shape(1) != m) {
        LightGBM::Log::Fatal("out array shape (%d, %d) is not compatible with the input (%d, %d)", out.shape(0),
                             out.shape(1), n, m);
    }
    num_threads = num_threads > 0 ? num_threads : 2 * omp_get_num_procs() - 1;
#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < n; i++) {
        std::map<std::string, double> temp = str_map(lines[i], sep1, sep2);
        for (size_t j = 0; j < m; j++) {
            auto it = temp.find(headers[j]);
            if (it == temp.end()) {
                mutable_at_(out, i, j) = default_value;
                empty_cnts[i] += 1;
            } else {
                mutable_at_(out, i, j) = it->second;
            }
        }
    }
    auto sum_of_elems = std::accumulate(empty_cnts.begin(), empty_cnts.end(), 0);
    LightGBM::Log::Info("#empty items:\t%d", sum_of_elems);
}

template <typename T>
py::array_t<T> fill_array_fast(const std::vector<std::vector<T>> &lines, py::array_t<T> &out, int st_idx,
                               T default_value, int num_threads = -1) {
    if (c_style(out) == false) { LightGBM::Log::Fatal("array must be c_style!"); }
    num_threads = num_threads > 0 ? num_threads : omp_get_num_procs();
#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < lines.size(); i++) {
        memcpy(&mutable_at_(out, st_idx + i), &lines[i][0], lines[i].size() * sizeof(T));
    }
    return std::move(out);
}

template <typename dtype, typename idtype, typename std::enable_if<std::is_integral<idtype>::value>::type * = nullptr>
std::unordered_map<idtype, py::array_t<dtype>>
groupby(const py::array_t<dtype> &data, py::array_t<idtype> &by,
        const std::string &agg = "mean" // mean, variance, max, min, count
        ,
        bool dropna = true) {
    if (c_style(data) == false || c_style(by) == false) { LightGBM::Log::Fatal("array must be c_style!"); }
    size_t len = (data.ndim() == 1 ? 1 : data.shape(1));
    auto uc_data = data.unchecked();
    size_t ids = 0;
    std::unordered_map<idtype, py::array_t<dtype>> map_out;
    if (agg == "count") {
        for (size_t i = 0; i < data.shape(0); i++) {
            auto idx_value = by.at(i);
            if (map_out.find(idx_value) == map_out.end()) { map_out[idx_value] = create_array({len}, dtype(0)); }

            if (uc_data.ndim() == 1 && (dropna == false || std::isnan(uc_data(i)) != true)) {
                mutable_at_(map_out[idx_value], 0) += 1;
            } else if (uc_data.ndim() == 2) {
                for (size_t j = 0; j < uc_data.shape(1); j++) {
                    if (dropna == false || std::isnan(uc_data(i, j)) != true) {
                        mutable_at_(map_out[idx_value], j) += 1;
                    }
                }
            }
        }
    } else if (agg == "mean") {
        std::unordered_map<idtype, py::array_t<dtype>> map_cnt;
        for (size_t i = 0; i < data.shape(0); i++) {
            auto idx_value = by.at(i);
            if (map_out.find(idx_value) == map_out.end()) {
                map_out[idx_value] = create_array({len}, dtype(0));
                map_cnt[idx_value] = create_array({len}, dtype(0));
            }

            if (uc_data.ndim() == 1 && std::isnan(uc_data(i)) != true) {
                mutable_at_(map_out[idx_value], 0) += at_(data, i);
                mutable_at_(map_cnt[idx_value], 0) += 1;
            } else if (uc_data.ndim() == 2) {
                for (size_t j = 0; j < uc_data.shape(1); j++) {
                    if (std::isnan(uc_data(i, j)) != true) {
                        mutable_at_(map_out[idx_value], j) += at_(data, i, j);
                        mutable_at_(map_cnt[idx_value], j) += 1;
                    }
                }
            }
        }
        for (auto it : map_out) {
            auto key = it.first;
            auto &sum = it.second;
            auto &cnt = map_cnt[key];
            for (size_t i = 0; i < map_cnt[key].shape(0); ++i) {
                if (cnt.at(i) > 0) {
                    mutable_at_(sum, i) /= at_(cnt, i);
                } else {
                    mutable_at_(sum, i) = std::numeric_limits<dtype>::quiet_NaN();
                }
            }
        }
    } else if (agg == "max" || agg == "min") {
        auto fn = (agg == "max") ? [](dtype a, dtype b) { return std::max(a, b); } :
                                   [](dtype a, dtype b) { return std::min(a, b); };
        for (size_t i = 0; i < data.shape(0); i++) {
            auto idx_value = by.at(i);
            if (map_out.find(idx_value) == map_out.end()) {
                map_out[idx_value] = create_array({len}, std::numeric_limits<dtype>::quiet_NaN());
            }

            if (uc_data.ndim() == 1 && (dropna == false || std::isnan(uc_data(i)) != true)) {
                mutable_at_(map_out[idx_value], 0) = fn(at_(map_out[idx_value], 0), at_(data, i));
            } else if (uc_data.ndim() == 2) {
                for (size_t j = 0; j < uc_data.shape(1); j++) {
                    if (dropna == false || std::isnan(uc_data(i, j)) != true) {
                        mutable_at_(map_out[idx_value], j) = fn(at_(map_out[idx_value], j), at_(data, i, j));
                    }
                }
            }
        }
    }

    return std::move(map_out);
}

void register_data_utils(py::module_ &m) {
    auto m_data = m.def_submodule("data", "data utils");
    m_data.def("string_to_dict", &parse_kv_str);

    m_data.def("fill_array_fast_float32", &fill_array_fast<float>);
    m_data.def("fill_array_fast_float64", &fill_array_fast<double>);
    m_data.def("fill_array_fast_int32", &fill_array_fast<int>);
    m_data.def("fill_array_fast_int", &fill_array_fast<int>);
    m_data.def("fill_array_fast_int64", &fill_array_fast<int64_t>);

    // regist json type
    py::class_<json11::Json>(m_data, "CppJson").def(py::init<>());
    m_data.def("json_from_str", [](const std::string &str) {
        std::string error;
        return json11::Json::parse(str, &error);
    });

    // regist group by operators
    m_data.def("groupby_double_int64", &groupby<double, int64_t>);
    m_data.def("groupby_double_uint64", &groupby<double, uint64_t>);
    m_data.def("groupby_double_int32", &groupby<double, int32_t>);
    m_data.def("groupby_double_uint32", &groupby<double, uint32_t>);
    m_data.def("groupby_float_int64", &groupby<float, int64_t>);
    m_data.def("groupby_float_uint64", &groupby<float, uint64_t>);
    m_data.def("groupby_float_int32", &groupby<float, int32_t>);
    m_data.def("groupby_float_uint32", &groupby<float, uint32_t>);
}

} // namespace gbct_utils
#endif //__DATA_UTILS_H__