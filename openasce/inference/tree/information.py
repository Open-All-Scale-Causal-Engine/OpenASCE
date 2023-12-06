#    Copyright 2023 AntGroup CO., Ltd.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


class CausalDataInfo(object):

    def __init__(self, conf, **kwargs):
        data_conf = conf.get('dataset', conf)
        self.n_treatment = data_conf.get('n_treatment')
        self.feature_columns = data_conf.get('feature', None)
        self.treatment_column = data_conf.get('treatment', None)
        self.feature_ratio = conf.get('feature_ratio', None)
        self.instance_ratio = conf.get('instance_ratio', None)
        self.n_period = data_conf.get('n_period')
        self.treat_dt = data_conf.get('treat_dt')

        hist_conf = conf.get('histogram', {})
        self.n_bins = hist_conf.get('max_bin_num', 64)
        self.min_point_per_bin = hist_conf.get('min_point_per_bin', 10)

        tree_conf = conf.get('tree', {})
        self.lambd = tree_conf.get('lambd', None)
        self.gamma = tree_conf.get('gamma', None)
        self.coef = tree_conf.get('coefficient', None)
        self.parallel_l2 = tree_conf.get('parallel_l2', None)

        self.min_point_num_node = tree_conf.get('min_point_num_node', None)
        self.max_depth = tree_conf.get('max_depth', None)
