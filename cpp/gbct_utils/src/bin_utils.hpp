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

void register_bin_utils(py::module_ &m) {
    auto m_bin = m.def_submodule("bin", "gbct bin");
    m_bin.def("FindBinParallel_double", &FindBinParallel<double>);
    m_bin.def("Value2BinParallel_double_int32", &Value2BinParallel<double, int32_t>);
    m_bin.def("Value2BinParallel_double_uint32", &Value2BinParallel<double, uint32_t>);
    m_bin.def("Value2BinParallel_float_int32", &Value2BinParallel<float, int32_t>);
    m_bin.def("Value2BinParallel_float_uint32", &Value2BinParallel<float, uint32_t>);

    py::class_<LightGBM::BinMapper>(m_bin, "BinMaper")
        .def(py::init<>())
        .def("GetUpperBoundValue", &LightGBM::BinMapper::GetUpperBoundValue)
        .def("is_trivial", &LightGBM::BinMapper::is_trivial)
        .def("BinToValue", &LightGBM::BinMapper::BinToValue)
        .def("MaxCatValue", &LightGBM::BinMapper::MaxCatValue)
        .def("SizesInByte", &LightGBM::BinMapper::SizesInByte)
        .def("ValueToBin", &LightGBM::BinMapper::ValueToBin)
        .def("GetDefaultBin", &LightGBM::BinMapper::GetDefaultBin)
        .def("GetMostFreqBin", &LightGBM::BinMapper::GetMostFreqBin)
        .def("bin_type", &LightGBM::BinMapper::bin_type)
        .def("bin_info_string", &LightGBM::BinMapper::bin_info_string)
        .def("sparse_rate", &LightGBM::BinMapper::sparse_rate)
        .def(py::pickle(
            [](const LightGBM::BinMapper &b) {
                return py::make_tuple(b.num_bin(), b.missing_type(), b.GetUpperBoundValue(), b.is_trivial(),
                                      b.sparse_rate(), b.bin_type(), b.categorical_2_bin(), b.bin_2_categorical(),
                                      b.min_val(), b.max_val(), b.GetDefaultBin(), b.GetMostFreqBin());
            },
            [](py::tuple t) { // __setstate_
                auto b = LightGBM::BinMapper(
                    t[0].cast<int>(), t[1].cast<LightGBM::MissingType>(), t[2].cast<std::vector<double>>(),
                    t[3].cast<bool>(), t[4].cast<double>(), t[5].cast<LightGBM::BinType>(),
                    t[6].cast<const std::unordered_map<int, unsigned int>>(), t[7].cast<const std::vector<int>>(),
                    t[8].cast<double>(), t[9].cast<double>(), t[10].cast<uint32_t>(), t[11].cast<uint32_t>());
                return b;
            }));

    py::class_<LightGBM::Bin>(m_bin, "Bin")
        .def(py::init(&LightGBM::Bin::CreateDenseBin))
        .def(py::init(&LightGBM::Bin::CreateSparseBin))
        .def("Push", &LightGBM::Bin::Push);

    py::enum_<LightGBM::MissingType>(m_bin, "MissingType")
        .value("None", LightGBM::MissingType::None)
        .value("Zero", LightGBM::MissingType::Zero)
        .value("NaN", LightGBM::MissingType::NaN)
        .export_values();

    py::enum_<LightGBM::BinType>(m_bin, "BinType")
        .value("NumericalBin", LightGBM::BinType::NumericalBin)
        .value("CategoricalBin", LightGBM::BinType::CategoricalBin)
        .export_values();
}
} // namespace gbct_utils