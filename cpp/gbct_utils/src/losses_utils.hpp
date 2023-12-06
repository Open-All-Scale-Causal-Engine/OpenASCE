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

#ifndef __LOSSES_UTILS_H__
#define __LOSSES_UTILS_H__

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "include/didtree_loss.h"
#include "include/causal_tree_loss.h"
#include "include/causal_tree_loss2.h"
#include "include/didtree_loss.h"

namespace py = pybind11;

namespace gbct_utils {

void register_losses_utils(py::module_ &m) {
    auto m_loss = m.def_submodule("splitting", "splitting loss utils");

    m_loss.def("gbct_splitting_loss_double_int32", &gbct_splitting_loss<double, int32_t>,
               py::return_value_policy::move);
    m_loss.def("gbct_splitting_loss_float64_int32", &gbct_splitting_loss<double, int32_t>,
               py::return_value_policy::move);

    m_loss.def("didtree_splitting_loss_double_int32", &didtree_splitting_loss<double, int32_t>,
               py::return_value_policy::move);
    m_loss.def("didtree_splitting_loss_float64_int32", &didtree_splitting_loss<double, int32_t>,
               py::return_value_policy::move);

    m_loss.def("causal_tree_splitting_loss_double_int32", &causal_tree_splitting_loss<double, int32_t>,
               py::return_value_policy::move);
    m_loss.def("causal_tree_splitting_loss_float64_int32", &causal_tree_splitting_loss<double, int32_t>,
               py::return_value_policy::move);

    m_loss.def("causal_tree_splitting_loss2_double_int32", &causal_tree_splitting_loss2<double, int32_t>,
               py::return_value_policy::move);
    m_loss.def("causal_tree_splitting_loss2_float64_int32", &causal_tree_splitting_loss2<double, int32_t>,
               py::return_value_policy::move);
}
} // namespace gbct_utils

#endif //__LOSSES_UTILS_H__