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

#ifndef __UTILS_H__
#define __UTILS_H__
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <execinfo.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "utils/log.h"

namespace py = pybind11;

namespace gbct_utils {

template <typename dtype>
inline py::array_t<dtype> create_array(const std::vector<size_t> &shape, dtype value = 0) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    std::vector<size_t> strides(shape.size());
    strides[shape.size() - 1] = sizeof(dtype);
    for (int i = shape.size() - 2; i > -1; --i) { strides[i] = shape[i + 1] * strides[i + 1]; }
    dtype *pdata = new dtype[size];
    memset(pdata, value, sizeof(dtype) * size);
    py::capsule free_post(pdata, [](void *f) {
        dtype *p = reinterpret_cast<dtype *>(f);
        DEBUG("freeing memory @%p", p);
        delete[] p;
    });
    return std::move(py::array_t<dtype>(shape, strides, pdata, free_post));
}

template <typename dtype>
py::array_t<dtype> create_array_from_address(const std::vector<size_t> &shape, const dtype *ptr,
                                             py::capsule *pfree = nullptr) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    std::vector<size_t> strides(shape.size());
    strides[shape.size() - 1] = sizeof(dtype);
    for (int i = shape.size() - 2; i > -1; --i) { strides[i] = shape[i + 1] * strides[i + 1]; }
    if (pfree == nullptr) {
        py::capsule free_post(ptr, [](void *f) {
            dtype *p = reinterpret_cast<dtype *>(f);
            DEBUG("Fake freeing memory @%p", p);
            // delete[] p;
        });
        return std::move(py::array_t<dtype>(shape, strides, ptr, free_post));
    } else {
        return std::move(py::array_t<dtype>(shape, strides, ptr, *pfree));
    }
}

/**
 * @brief copy array from `src` to `dec`
 *
 * @param des
 * @param src
 * @param n The number of elements copied from des to src. n=-1 means to copy as many elements as possible.
 */
template <typename dtype, int r>
void array_copy(py::array_t<dtype, r> &des, const py::array_t<dtype, r> &src, int n = -1) {
    // check strides
    for (size_t i = 0; i < des.ndim(); i++) {
        if (des.strides(i) != src.strides(i)) {
            LightGBM::Log::Fatal("the strides of src and des are not compatible!");
        }
    }
    n = (n >= 0 && n <= std::min(src.size(), des.size()) ? n : std::min(src.size(), des.size()));
    memcpy(des.mutable_data(), src.data(), sizeof(dtype) * n);
}

/**
 * @brief copy array from `src` to `dec`. Be caution, without check compatibility.
 *
 * @tparam dtype
 * @param des
 * @param src
 * @param n The number of elements copied from des to src. n=-1 means to copy as many elements as possible.
 */
template <typename dtype>
void array_copy(dtype *des, const py::array_t<dtype> &src, int n = -1) {
    n = (n >= 0 && n <= src.size() ? n : src.size());
    memcpy(des, src.data(), sizeof(dtype) * n);
}

/**
 * @brief judge the array is column major or not
 *
 * @param arr
 */
template <typename dtype>
bool f_style(const py::array_t<dtype> &arr) {
    // test f style
    int cur = arr.itemsize();
    for (size_t i = 0; i < arr.ndim() - 1; ++i) {
        if (arr.strides(i) != cur) { return false; }
        cur = cur * arr.shape(i);
    }
    return true;
}

template <typename dtype>
bool c_style(const py::array_t<dtype> &arr) {
    // test c style
    int cur = arr.itemsize();
    for (size_t i = arr.ndim() - 1; i > 0; --i) {
        if (arr.strides(i) != cur) { return false; }
        cur = cur * arr.shape(i);
    }
    return true;
}

template <typename dtype1, typename dtype2>
int shape_match(const py::array_t<dtype1> &a, const py::array_t<dtype2> &b, size_t axis = 0) {
    if (a.ndim() <= axis || b.ndim() <= axis) {
        LightGBM::Log::Fatal("The axis(%d) larger than the dimension of matrix(%d or %d)", axis, a.ndim(), b.ndim());
        return -1;
    }
    return a.shape(axis) == b.shape(axis) ? 1 : 0;
}

template <typename dtype1, typename dtype2>
int shapes_match(const py::array_t<dtype1> &a, const py::array_t<dtype2> &b) {
    for (size_t i = 0; i < a.ndim(); i++) {
        if (shape_match(a, b, i) <= 0) {
            LightGBM::Log::Fatal("all the input array dimensions for the concatenation axis must match exactly, "
                                 "but along dimension %d, the first array has size %d and the second array has size %d",
                                 i, a.shape(i), b.shape(i));
        }
    }
    return 0;
}

template <typename dtype>
inline bool empty_array(const py::array_t<dtype> &v) {
    return (v.size() == 0);
}

template <typename dtype, typename... Ix>
dtype *mutable_data_(py::array_t<dtype> &data, Ix... index) {
    auto n_dim = data.ndim();
    auto n_param = sizeof...(index);
    if (n_dim == (sizeof...(index))) {
        return data.mutable_data(index...);
    } else if (n_dim == (sizeof...(index)) + 1) {
        return data.mutable_data(index..., 0);
    } else if (n_dim == (sizeof...(index)) + 2) {
        return data.mutable_data(index..., 0, 0);
    }
    LightGBM::Log::Fatal("mutable_data_: index dimension mismatch (%d not equal to %d) or the missed index more than 2",
                         n_dim, sizeof...(index));
    return static_cast<dtype *>(nullptr);
}

template <typename dtype, typename... Ix>
const dtype *data_(const py::array_t<dtype> &data, Ix... index) {
    auto n_dim = data.ndim();
    auto n_param = sizeof...(index);
    if (n_dim == (sizeof...(index))) {
        return data.data(index...);
    } else if (n_dim == (sizeof...(index)) + 1) {
        return data.data(index..., 0);
    } else if (n_dim == (sizeof...(index)) + 2) {
        return data.data(index..., 0, 0);
    }
    LightGBM::Log::Fatal("data_: index dimension mismatch (%d not equal to %d) or the missed index more than 2", n_dim,
                         sizeof...(index));
    return static_cast<dtype *>(nullptr);
}

template <typename dtype, typename... Ix>
dtype &mutable_at_(py::array_t<dtype> &data, Ix... index) {
    auto n_dim = data.ndim();
    if (n_dim == (sizeof...(index))) {
        return data.mutable_at(index...);
    } else if (n_dim == (sizeof...(index)) + 1) {
        return data.mutable_at(index..., 0);
    } else if (n_dim == (sizeof...(index)) + 2) {
        return data.mutable_at(index..., 0, 0);
    }
    int size = 16;
    void *array[16];
    int stack_num = backtrace(array, size);
    char **stacktrace = backtrace_symbols(array, stack_num);
    for (int i = 0; i < stack_num; ++i) { printf("%s\n", stacktrace[i]); }
    free(stacktrace);
    LightGBM::Log::Fatal("[%p]index dimension mismatch (%d not equal to %d) or the missed index more than 2",
                         __builtin_return_address(1), n_dim, sizeof...(index));
    return *static_cast<dtype *>(nullptr);
}

template <typename dtype, typename... Ix>
const dtype &at_(const py::array_t<dtype> &data, Ix... index) {
    auto n_dim = data.ndim();
    if (n_dim == (sizeof...(index))) {
        return data.at(index...);
    } else if (n_dim == (sizeof...(index) + 1)) {
        return data.at(index..., 0);
    } else if (n_dim == (sizeof...(index) + 2)) {
        return data.at(index..., 0, 0);
    }
    LightGBM::Log::Fatal("at_: index dimension mismatch (%d not equal to %d) or the missed index more than 2", n_dim,
                         sizeof...(index));
    return *static_cast<dtype *>(nullptr);
}

template <typename dtype>
void print_mat(const py::array_t<dtype> &data) {
    size_t n = data.shape(0), m = data.shape(1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) { std::cerr << data.at(i, j) << ' '; }
        std::cerr << std::endl;
    }
}
} // namespace gbct_utils
#endif //__UTILS_H__
