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

#ifndef __NPARRAY_OPS__
#define __NPARRAY_OPS__

#include <cmath>
#include <iostream>
#include <functional>
#include <memory>
#include <omp.h>
#include <unordered_map>
#include <type_traits>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

namespace gbct_utils {

template <typename dtype>
void binary_op(const py::array_t<dtype> &a, const py::array_t<dtype> &b, py::array_t<dtype> &o, int a_offset,
               int b_offset, int o_offset, int dim, std::function<dtype(dtype, dtype)> bin_func) {
    auto pa = a.data() + a_offset, pb = b.data() + b_offset, po = o.mutable_data() + o_offset;
    if (dim + 1 == o.ndim()) { // last dimension
        if (a.shape(dim) == 1) {
            for (size_t i = 0; i < o.shape(dim); i++) po[i] = bin_func(pa[0], pb[i]);
        } else if (b.shape(dim) == 1) {
            for (size_t i = 0; i < o.shape(dim); i++) po[i] = bin_func(pa[i], pb[0]);
        } else {
            for (size_t i = 0; i < o.shape(dim); i++) po[i] = bin_func(pa[i], pb[i]);
        }
        return;
    }
    for (size_t i = 0; i < o.shape(dim); i++) {
        binary_op(a, b, o, a_offset, b_offset, o_offset, dim + 1, bin_func);
        if (a.shape(dim) > 1) { a_offset += a.strides(dim) / sizeof(dtype); }
        if (b.shape(dim) > 1) { b_offset += b.strides(dim) / sizeof(dtype); }
        o_offset += o.strides(dim) / sizeof(dtype);
    }
}

template <typename dtype>
py::array_t<dtype> operator+(const py::array_t<dtype> &A, const py::array_t<dtype> &B) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    for (size_t i = 0; i < A.ndim(); i++) {
        if (A.shape(i) != B.shape(i) && B.shape(i) > 1 && A.shape(i) > 1) {
            LightGBM::Log::Fatal("all the input array dimensions for the concatenation axis must match exactly, "
                                 "but along dimension %d, the first array has size %d and the second array has size %d",
                                 i, A.shape(i), A.shape(i));
        }
        shape[i] = std::max(A.shape(i), B.shape(i));
    }

    py::array_t<dtype> out(shape);
    std::function<dtype(dtype, dtype)> add_func = [](dtype a, dtype b) { return a + b; };
    binary_op(A, B, out, 0, 0, 0, 0, add_func);
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator+(const py::array_t<dtype> &A, dtype B) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    py::array_t<dtype> out(shape, A.data());
    auto ptr = out.mutable_data();
    for (size_t i = 0; i < A.size(); i++) { ptr[i] += B; }
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator+(dtype B, const py::array_t<dtype> &A) {
    return A + B;
};

template <typename dtype>
py::array_t<dtype> operator-(const py::array_t<dtype> &A, const py::array_t<dtype> &B) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    for (size_t i = 0; i < A.ndim(); i++) {
        if (A.shape(i) != B.shape(i) && B.shape(i) > 1 && A.shape(i) > 1) {
            LightGBM::Log::Fatal("all the input array dimensions for the concatenation axis must match exactly, "
                                 "but along dimension %d, the first array has size %d and the second array has size %d",
                                 i, A.shape(i), A.shape(i));
        }
        shape[i] = std::max(A.shape(i), B.shape(i));
    }

    py::array_t<dtype> out(shape);
    std::function<dtype(dtype, dtype)> minus_func = [](dtype a, dtype b) { return a - b; };
    binary_op(A, B, out, 0, 0, 0, 0, minus_func);
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator-(const py::array_t<dtype> &A, dtype B) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    py::array_t<dtype> out(shape, A.data());
    auto ptr = out.mutable_data();
    for (size_t i = 0; i < A.size(); i++) { ptr[i] -= B; }
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator-(dtype B, const py::array_t<dtype> &A) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    py::array_t<dtype> out(shape, A.data());
    auto ptr = out.mutable_data();
    for (size_t i = 0; i < A.size(); i++) { ptr[i] = B - ptr[i]; }
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator-(const py::array_t<dtype> &A) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    py::array_t<dtype> out(shape);
    auto ptr = out.mutable_data();
    auto pdata = A.data();
    for (size_t i = 0; i < A.size(); i++) { ptr[i] = -pdata[i]; }
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator/(const py::array_t<dtype> &A, dtype B) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    py::array_t<dtype> out(shape, A.data());
    auto ptr = out.mutable_data();
    for (size_t i = 0; i < A.size(); i++) { ptr[i] /= B; }
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator/(const py::array_t<dtype> &A, const py::array_t<dtype> &B) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    for (size_t i = 0; i < A.ndim(); i++) {
        if (A.shape(i) != B.shape(i) && B.shape(i) > 1 && A.shape(i) > 1) {
            LightGBM::Log::Fatal("all the input array dimensions for the concatenation axis must match exactly, "
                                 "but along dimension %d, the first array has size %d and the second array has size %d",
                                 i, A.shape(i), A.shape(i));
        }
        shape[i] = std::max(A.shape(i), B.shape(i));
    }

    py::array_t<dtype> out(shape);
    std::function<dtype(dtype, dtype)> div_func = [](dtype a, dtype b) { return a / b; };
    binary_op(A, B, out, 0, 0, 0, 0, div_func);
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator/(dtype B, const py::array_t<dtype> &A) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    py::array_t<dtype> out(shape, A.data());
    auto ptr = out.mutable_data();
    for (size_t i = 0; i < A.size(); i++) { ptr[i] = B / ptr[i]; }
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator*(const py::array_t<dtype> &A, const py::array_t<dtype> &B) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    for (size_t i = 0; i < A.ndim(); i++) {
        if (A.shape(i) != B.shape(i) && B.shape(i) > 1 && A.shape(i) > 1) {
            LightGBM::Log::Fatal("all the input array dimensions for the concatenation axis must match exactly, "
                                 "but along dimension %d, the first array has size %d and the second array has size %d",
                                 i, A.shape(i), A.shape(i));
        }
        shape[i] = std::max(A.shape(i), B.shape(i));
    }

    py::array_t<dtype> out(shape);
    std::function<dtype(dtype, dtype)> mul_func = [](dtype a, dtype b) { return a * b; };
    binary_op(A, B, out, 0, 0, 0, 0, mul_func);
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator*(const py::array_t<dtype> &A, dtype B) {
    auto shape = std::vector<size_t>(A.shape(), A.shape() + A.ndim());
    py::array_t<dtype> out(shape, A.data());
    auto ptr = out.mutable_data();
    for (size_t i = 0; i < A.size(); i++) { ptr[i] *= B; }
    return std::move(out);
};

template <typename dtype>
py::array_t<dtype> operator*(dtype B, const py::array_t<dtype> &A) {
    return A * B;
};

// sum
template <typename dtype>
dtype sum(const py::array_t<dtype> &m) {
    if (c_style(m) == false) { FATAL("f_style is not supported"); }
    dtype out = 0;
    auto pdata = m.data();
    for (size_t i = 0; i < m.size(); i++) { out += pdata[i]; }
    return std::move(out);
}

template <typename dtype>
dtype sum(const dtype *m, const std::vector<size_t> &shape) {
    dtype out = 0;
    auto n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    return std::accumulate(m, m + n, 0);
}

template <typename dtype>
dtype sum(const std::unique_ptr<dtype[]> &m, const std::vector<size_t> &shape) {
    dtype out = 0;
    size_t counts = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    for (size_t i = 0; i < counts; i++) { out += m[i]; }
    return out;
}

template <typename dtype>
inline py::array_t<dtype> sum(const py::array_t<dtype> &m, int axis, bool keep_dims = false) {
    if (c_style(m) == false) { FATAL("f_style is not supported"); }
    if (m.ndim() <= axis) { FATAL("axis %d is out of bounds for array of dimension %d", axis, m.ndim()); }
    std::vector<size_t> out_shape(m.shape(), m.shape() + m.ndim());
    out_shape[axis] = 1;
    size_t size = std::accumulate(m.shape(), m.shape() + m.ndim(), 1, std::multiplies<size_t>());
    std::vector<size_t> strides(m.strides(), m.strides() + m.ndim());
    dtype *pout = new dtype[size];
    py::capsule free_post(pout, [](void *f) {
        dtype *p = reinterpret_cast<dtype *>(f);
        DEBUG("freeing C++ memory @%p", p);
        delete[] p;
    });
    memset(pout, dtype(0), size * sizeof(dtype));

    const dtype *pdata = static_cast<const dtype *>(m.data());
    size_t up_stride = strides[axis] / sizeof(dtype) * m.shape(axis);
    size_t low_stride = strides[axis] / sizeof(dtype);
    for (size_t i = 0; i < m.size(); i++) {
        size_t i_ = std::floor(i / up_stride) * up_stride / m.shape(axis) + (i % low_stride);
        pout[i_] += pdata[i];
    }
    if (keep_dims == false) {
        out_shape.erase(out_shape.begin() + axis);
        strides.erase(strides.begin() + axis);
    }
    for (size_t i = 0; i < axis; i++) { strides[i] /= m.shape(axis); }
    return std::move(py::array_t<dtype>(out_shape, strides, pout, free_post));
}

template <typename dtype>
py::array_t<dtype> sum(const dtype *m, const std::vector<size_t> &shape, int axis, bool keep_dims = false) {
    return std::move(sum(create_array_from_address(shape, m), axis, keep_dims));
}

// mean
template <typename dtype>
dtype mean(const py::array_t<dtype> &m, int axis) {
    return sum(m) / m.shape(axis);
}

template <typename dtype>
py::array_t<dtype> mean(const py::array_t<dtype> &m, int axis, bool keep_dims = false) {
    return std::move(sum(m, axis, keep_dims) / m.shape(axis));
}

template <typename dtype>
dtype *_concatenate(const py::array_t<dtype> &A, const py::array_t<dtype> &B, dtype *out, int A_off, int B_off,
                    int cur_axis, int axis) {
    if (cur_axis == axis) {
        auto nbytes = A.strides(axis) * A.shape(axis);
        memcpy(out, A.data() + A_off, nbytes);
        memcpy(out + nbytes, B.data() + B_off, B.strides(axis) * B.shape(axis));
        return out + B.strides(axis) * (B.shape(axis) + A.shape(axis));
    } else {
        for (size_t i = 0; i < A.shape(cur_axis); i++) {
            out = _concatenate(A, B, out, A_off, B_off, cur_axis + 1, axis);
            A_off += A.strides(cur_axis) / sizeof(dtype);
            B_off += B.strides(cur_axis) / sizeof(dtype);
        }
    }
    return std::move(out);
}

template <typename dtype>
py::array_t<dtype> concatenate(const py::array_t<dtype> &A, const py::array_t<dtype> &B, int axis) {
    if (A.ndim() != B.ndim()) { LightGBM::Log::Fatal("all the input array dimensions must be the same length."); }
    std::vector<size_t> out_shape(A.shape(), A.shape() + A.ndim());
    shapes_match(A, B);
    out_shape[axis] += B.shape(axis);

    py::array_t<dtype> out = py::array_t<dtype>(out_shape);
    auto stride = out.strides(axis) / sizeof(dtype);
    auto pout = out.mutable_data();

    _concatenate(A, B, pout, 0, 0, 0, axis);
    return std::move(out);
}

} // namespace gbct_utils

#endif //__NPARRAY_OPS__