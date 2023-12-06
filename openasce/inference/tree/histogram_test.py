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

import unittest as ut

from pyhocon import ConfigFactory

from openasce.inference.tree.histogram import Histogram


class TestHistogram(ut.TestCase):
    def test_histogram(self):
        conf = ConfigFactory.from_dict(
            {
                "dataset": {
                    "n_treatment": 2,
                    "treat_dt": 0,
                    "treatment": "treatment",
                    "n_period": 1,
                },
                "histogram": {"max_bin_num": 2, "min_point_per_bin": 1},
            }
        )
        hist = Histogram(conf)
