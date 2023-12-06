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

#include <pybind11/pybind11.h>

#include "include/did_utils.h"
#include "include/gbct_utils.h"
#include "include/utils/nparray_ops.h"

namespace py = pybind11;

void gbct_utils::DebiasNode::set_outcomes(const py::array_t<double> &value) {
    outcomes = create_array<double>({size_t(value.shape(0)), size_t(value.shape(1))});
    array_copy(outcomes, value);
}

inline const py::array_t<double> &gbct_utils::DebiasNode::get_info(const std::string &name) const {
    if (has(name) == false) { LightGBM::Log::Fatal("The `name`(%s) is not included in DebiasNode", name.c_str()); }
    if (name == "outcomes") {
        return outcomes;
    } else if (name == "bias") {
        return bias;
    } else if (name == "eta") {
        return eta;
    } else if (name == "debiased_effect") {
        return debiased_effect;
    } else if (name == "effect") {
        return effect;
    }
    if (infos.find(name) == infos.end()) { LightGBM::Log::Fatal("The key `%s` is out of range", name.c_str()); }
    return infos.at(name);
}

int gbct_utils::DebiasNode::set_info(const std::string &name, const py::array_t<double> &value) {
    std::vector<size_t> shape(value.shape(), value.shape() + value.ndim());
    if (name == "outcomes") {
        outcomes = create_array(shape, 0.0);
        array_copy(outcomes, value);
        // insert `effect`
        infos["effect"] = create_array(shape, double(0));
        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                mutable_at_(infos["effect"], i, j) = at_(outcomes, i, j) - at_(outcomes, i, 0);
            }
        }
    } else if (name == "bias") {
        bias = create_array(shape, 0.0);
        array_copy(bias, value);
    } else if (name == "eta") {
        eta = create_array(shape, 0.0);
        array_copy(eta, value);
    } else if (name == "debiased_effect") {
        debiased_effect = create_array(shape, 0.0);
        array_copy(debiased_effect, value);
    } else if (name == "effect") {
        effect = create_array(shape, 0.0);
        array_copy(effect, value);
    }
    infos[name] = create_array(shape, 0.0);
    array_copy(infos[name], value);
    return 0;
}
