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

#include "include/gbct_utils.h"
#include "include/agg_fn.h"
#include "include/did_utils.h"
#include "include/gbct_bin.h"
#include "include/bin.h"
#include "include/predict.h"
#include "src/dense_bin.hpp"
#include "src/data_utils.hpp"
#include "src/bin_utils.hpp"
#include "src/losses_utils.hpp"


using namespace gbct_utils;

PYBIND11_MODULE(gbct_utils, m)
{
    m.doc() = "pybind11 for gbct"; // optional module docstring

    {
        auto m_comm = m.def_submodule("common", "gbct common utils");

        m_comm.def("sumby_double_int32", &sumby_nogil<double, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_float_int32", &sumby_nogil<float, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_float_int64", &sumby_nogil<float, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_int32_int32", &sumby_nogil<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_int64_int64", &sumby_nogil<int64_t, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_int32_int64", &sumby_nogil<int32_t, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_int64_int32", &sumby_nogil<int64_t, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_float_uint32", &sumby_nogil<float, uint32_t>, py::return_value_policy::move);

        m_comm.def("sumby_parallel_double_int32", &sumby_parallel<double, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_float_int32", &sumby_parallel<float, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_float_int64", &sumby_parallel<float, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_int32_int32", &sumby_parallel<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_int64_int64", &sumby_parallel<int64_t, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_int32_int64", &sumby_parallel<int32_t, int64_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_int64_int32", &sumby_parallel<int64_t, int32_t>, py::return_value_policy::move);
        m_comm.def("sumby_parallel_float_uint32", &sumby_parallel<float, uint32_t>, py::return_value_policy::move);

        m_comm.def("countby_double_uint32", &countby_nogil<double, uint32_t>, py::return_value_policy::move);
        m_comm.def("countby_double_int32", &countby_nogil<double, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_double_int64", &countby_nogil<double, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_double_uint64", &countby_nogil<double, uint64_t>, py::return_value_policy::move);
        m_comm.def("countby_float_int64", &countby_nogil<float, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_float_uint64", &countby_nogil<float, uint64_t>, py::return_value_policy::move);
        m_comm.def("countby_float_int32", &countby_nogil<float, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_float_uint32", &countby_nogil<float, uint32_t>, py::return_value_policy::move);
        m_comm.def("countby_int32_int32", &countby_nogil<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_int64_int64", &countby_nogil<int64_t, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_int32_int64", &countby_nogil<int32_t, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_int64_int32", &countby_nogil<int64_t, int32_t>, py::return_value_policy::move);


        m_comm.def("countby_parallel_double_uint32", &countby_parallel<double, uint32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_double_int32", &countby_parallel<double, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_double_int64", &countby_parallel<double, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_double_uint64", &countby_parallel<double, uint64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_float_int64", &countby_parallel<float, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_float_uint64", &countby_parallel<float, uint64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_float_int32", &countby_parallel<float, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_float_uint32", &countby_parallel<float, uint32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_int32_int32", &countby_parallel<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_int64_int64", &countby_parallel<int64_t, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_int32_int64", &countby_parallel<int32_t, int64_t>, py::return_value_policy::move);
        m_comm.def("countby_parallel_int64_int32", &countby_parallel<int64_t, int32_t>, py::return_value_policy::move);

        m_comm.def("indexbyarray_double_int32", &indexbyarray<double, int32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_double_uint32", &indexbyarray<double, uint32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_double_int64", &indexbyarray<double, int64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_double_uint64", &indexbyarray<double, uint64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_float_int32", &indexbyarray<float, int32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_float_uint32", &indexbyarray<float, uint32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_float_int64", &indexbyarray<float, int64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray_float_uint64", &indexbyarray<float, uint64_t>, py::return_value_policy::move);

        m_comm.def("indexbyarray2_double_int32", &indexbyarray2<double, int32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_double_uint32", &indexbyarray2<double, uint32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_double_int64", &indexbyarray2<double, int64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_double_uint64", &indexbyarray2<double, uint64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_float_int32", &indexbyarray2<float, int32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_float_uint32", &indexbyarray2<float, uint32_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_float_int64", &indexbyarray2<float, int64_t>, py::return_value_policy::move);
        m_comm.def("indexbyarray2_float_uint64", &indexbyarray2<float, uint64_t>, py::return_value_policy::move);

        m_comm.def("update_x_map_int32", &update_x_map<int32_t>, py::return_value_policy::move);
        m_comm.def("update_histogram_double_int32", &update_histogram<double, int32_t>, py::return_value_policy::move);
        m_comm.def("update_histogram_int32_int32", &update_histogram<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("update_histogram_float32_int32", &update_histogram<float, int32_t>, py::return_value_policy::move);
        m_comm.def("update_histograms_double_int32", &update_histograms<double, int32_t>, py::return_value_policy::move);
        m_comm.def("update_histograms_int32_int32", &update_histograms<int32_t, int32_t>, py::return_value_policy::move);
        m_comm.def("update_histograms_float32_int32", &update_histograms<float, int32_t>, py::return_value_policy::move);

        py::class_<Node>(m_comm, "CppNode")
            // .def(py::init<>())
            .def_readwrite("split_feature", &Node::split_feature)
            .def_readwrite("split_thresh", &Node::split_thresh)
            .def_property("children", &Node::get_children, &Node::set_children)
            .def_readwrite("is_leaf", &Node::is_leaf)
            .def_readwrite("leaf_id", &Node::leaf_id)
            .def_readwrite("level_id", &Node::level_id)
            .def("get_property_keys", &Node::get_property_keys);

        py::class_<DebiasNode, Node>(m_comm, "CppDebiasNode")
            .def(py::init<>())
            // .def_readwrite("outcomes", &DebiasNode::outcomes)
            .def_property("outcomes", &DebiasNode::get_outcomes, &DebiasNode::set_outcomes)
            .def_readwrite("bias", &DebiasNode::bias)
            .def_readwrite("eta", &DebiasNode::eta)
            .def_readwrite("debiased_effect", &DebiasNode::debiased_effect)
            .def_readwrite("effect", &DebiasNode::effect)
            .def("get_property_keys", &DebiasNode::get_property_keys)
            .def("get_info", &DebiasNode::get_info)
            .def("set_info", &DebiasNode::set_info)
            .def(py::pickle([](const DebiasNode& b) {
            return b.dump();
                }, [](py::tuple t) {// __setstate_
                    return DebiasNode::load(t);
                    }));

        py::class_<DiDNode, DebiasNode>(m_comm, "CppDiDNode")
            .def("get_info", &DiDNode::get_info)
            .def("set_info", &DiDNode::set_info)
            .def(py::init<>())
            .def("get_property_keys", &DiDNode::get_property_keys);

        m_comm.def("predict_debias", &predict<DebiasNode>, py::return_value_policy::move);
        m_comm.def("predict_did", &predict<DiDNode>, py::return_value_policy::move);

        // m_comm.def("gbct_splitting_losses", &gbct_splitting_losses<double, int32_t>, py::return_value_policy::move);
        // m_comm.def("causal_tree_splitting_losses", &causal_tree_splitting_losses<double, int32_t>, py::return_value_policy::move);
        // m_comm.def("didtree_splitting_losses", &didtree_splitting_losses<double, int32_t>, py::return_value_policy::move);
        m_comm.def("sum", static_cast<py::array_t<double>(*)(const py::array_t<double> &, int, bool)>(&sum<double>));
        m_comm.def("concatenate", &concatenate<double>);
        // just for unittest
        m_comm.def("array_add", [](const py::array_t<double>& a, const py::array_t<double>& b) {return a + b;});
        m_comm.def("array_mul", [](const py::array_t<double>& a, const py::array_t<double>& b) {return a * b;});

        m_comm.def("reset_loglevel", LightGBM::Log::ResetLogLevel);

        py::enum_<LightGBM::LogLevel>(m_comm, "LogLevel")
            .value("Debug", LightGBM::LogLevel::Debug)
            .value("Fatal", LightGBM::LogLevel::Fatal)
            .value("Info", LightGBM::LogLevel::Info)
            .value("Trace", LightGBM::LogLevel::Trace)
            .value("Warning", LightGBM::LogLevel::Warning)
            .export_values();

        register_data_utils(m_comm);
        register_bin_utils(m);
        register_losses_utils(m);
    }
}
