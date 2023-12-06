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

#ifndef __GBCT_BIN_H__
#define __GBCT_BIN_H__
#include <omp.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <iostream>
#include <vector>

#include "bin.h"
#include "utils.h"
#include "utils/common.h"
#include "utils/log.h"
#include "utils/threading.h"

namespace py = pybind11;

namespace gbct_utils {
template <typename dtype>
std::vector<LightGBM::BinMapper>
FindBinParallel(const py::array_t<dtype> &data, int max_bin = 64, int min_data_in_bin = 100, int min_split_data = 100,
                bool pre_filter = false, int bin_type = LightGBM::BinType::NumericalBin, bool use_missing = true,
                bool zero_as_missing = false, const std::vector<double> &forced_upper_bounds = {}) {
    auto shape = data.shape();
    size_t nfeat = ((data.ndim() == 2) ? shape[1] : 1);
    size_t ninst = shape[0];
    std::vector<LightGBM::BinMapper> mappers(nfeat);
    std::vector<std::vector<dtype>> nzero_data(nfeat);
    int num_threads = OMP_NUM_THREADS();
    LightGBM::Log::Info("data(%d, %d) will be used to binning", shape[0], shape[1]);
    py::gil_scoped_release release;
    omp_set_num_threads(num_threads);
#pragma omp parallel for
    for (size_t i = 0; i < nfeat; i++) {
        for (size_t j = 0; j < ninst; j++) {
            if (data.at(j, i) != dtype(0)) nzero_data[i].push_back(data.at(j, i));
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < nfeat; i++) {
        mappers[i].FindBin(nzero_data[i].data(), nzero_data[i].size(), ninst, max_bin, min_data_in_bin, min_split_data,
                           pre_filter, LightGBM::BinType(bin_type), use_missing, zero_as_missing, forced_upper_bounds);
    }

    return mappers;
}

template <typename dtype, typename idtype, typename std::enable_if<std::is_integral<idtype>::value>::type * = nullptr>
py::array_t<idtype> &Value2BinParallel(const py::array_t<dtype> &data, const std::vector<LightGBM::BinMapper> &mappers,
                                       py::array_t<idtype> &out, int num_threads = -1) {
    size_t nfeat = mappers.size();
    size_t ninst = data.shape(0);
    size_t block_size = 10240;
    if (nfeat != data.shape(1)) LightGBM::Log::Fatal("data(%d) and mappers(%d) don't match", data.shape(0), nfeat);

    if ((out.shape(0) != ninst) || (out.shape(1) != data.shape(1)))
        LightGBM::Log::Fatal("data(%d, %d) and out(%d, %d) don't match", data.shape(0), data.shape(0), out.shape(0),
                             out.shape(0));
    OMP_SET_NUM_THREADS(num_threads);
#pragma omp parallel for
    for (size_t i = 0; i < nfeat; i++) {
        for (size_t j = 0; j < ninst; j++) { out.mutable_at(j, i) = mappers[i].ValueToBin(data.at(j, i)); }
    }

    return out;
}
} // namespace gbct_utils
#endif // end for __GBCT_BIN_H__