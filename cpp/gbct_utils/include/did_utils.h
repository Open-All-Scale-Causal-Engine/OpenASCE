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

#ifndef __DID_HEADER__
#define __DID_HEADER__

#include <cmath>
#include <iostream>
#include <set>
#include <unordered_map>

#include <omp.h>
#include <pybind11/numpy.h>

#include "include/utils.h"
#include "utils/nparray_ops.h"

namespace py = pybind11;

namespace gbct_utils {

class Node {
    std::set<std::string> _info_keys = {"split_feature", "split_thresh", "leaf_id", "level_id", "children", "is_leaf"};

public:
    int split_feature;
    double split_thresh;
    int leaf_id;
    int level_id;
    int children[2];
    bool is_leaf;
    void set_children(const std::vector<int> &_ch) {
        memcpy(children, _ch.data(), 2 * sizeof(int));
    }
    inline int get_split_feature() const {
        return split_feature;
    }
    inline double get_split_thresh() const {
        return split_thresh;
    }
    inline int get_leaf_id() const {
        return leaf_id;
    }
    inline int get_level_id() const {
        return level_id;
    }
    inline int leaf() const {
        return is_leaf;
    }
    std::vector<int> get_children() const {
        return std::vector<int>(children, children + 2);
    }
    virtual const py::array_t<double> &get_info(const std::string &name) const = 0;
    virtual int set_info(const std::string &name, const py::array_t<double> &) = 0;
    virtual bool has(const std::string &key) const {
        return _info_keys.find(key) != _info_keys.end();
    }
    virtual std::vector<std::string> get_property_keys() const {
        return std::vector<std::string>(_info_keys.begin(), _info_keys.end());
    }
};

class DebiasNode : public Node {
protected:
    std::set<std::string> _info_keys = {"outcomes", "bias", "effect", "eta", "debiased_effect"};
    std::unordered_map<std::string, py::array_t<double>> infos;

public:
    py::array_t<double> bias;
    py::array_t<double> outcomes;
    py::array_t<double> effect;
    py::array_t<double> eta;
    py::array_t<double> debiased_effect;
    virtual const py::array_t<double> &get_outcomes() const {
        return outcomes;
    }
    virtual void set_outcomes(const py::array_t<double> &);
    virtual bool has(const std::string &key) const {
        return (this->Node::has(key)) || (_info_keys.find(key) != _info_keys.end()) || (infos.find(key) != infos.end());
    }
    virtual const py::array_t<double> &get_info(const std::string &name) const;
    virtual int set_info(const std::string &name, const py::array_t<double> &);
    virtual std::vector<std::string> get_property_keys() const {
        auto tmp = this->Node::get_property_keys();
        tmp.insert(tmp.end(), _info_keys.begin(), _info_keys.end());
        return std::move(tmp);
    }
    // override
    py::tuple dump() const {
        return py::make_tuple(split_feature, split_thresh, leaf_id, level_id, get_children(), is_leaf, bias, outcomes,
                              effect, eta, debiased_effect, infos, _info_keys);
    }
    static DebiasNode load(py::tuple p) {
        auto o = DebiasNode();
        o.split_feature = p[0].cast<int>();
        o.split_thresh = p[1].cast<double>();
        o.leaf_id = p[2].cast<int>();
        o.level_id = p[3].cast<int>();
        o.set_children(p[4].cast<std::vector<int>>());
        o.is_leaf = p[5].cast<bool>();
        o.set_info("bias", p[6].cast<py::array_t<double>>());
        o.set_outcomes(p[7].cast<py::array_t<double>>());
        o.set_info("effect", p[8].cast<py::array_t<double>>());
        o.set_info("eta", p[9].cast<py::array_t<double>>());
        o.set_info("debiased_effect", p[10].cast<py::array_t<double>>());
        o.infos = p[11].cast<std::unordered_map<std::string, py::array_t<double>>>();
        o._info_keys = p[12].cast<std::set<std::string>>();
        return o;
    }
};

class DiDNode : public DebiasNode {
    std::set<std::string> _info_keys = {"outcomes", "bias", "zz", "zy"};

public:
    virtual bool has(const std::string &key) const {
        return (this->DebiasNode::has(key)) || (_info_keys.find(key) != _info_keys.end());
    }
    virtual std::vector<std::string> get_property_keys() const {
        auto tmp = this->DebiasNode::get_property_keys();
        tmp.insert(tmp.end(), _info_keys.begin(), _info_keys.end());
        return tmp;
    }
};

} // namespace gbct_utils
#endif // __DID_HEADER__