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

from unittest import TestCase

from openasce.discovery.regression_discovery.regression_discovery import (
    CausalRegressionDiscovery,
)
from openasce.utils.logger import logger


class TestSearchDiscovery(TestCase):
    def setUp(self) -> None:
        pass

    def test_regression_discovery(self) -> None:
        from tests.datasets.continuous_search_data import ContinuousSearchData

        m = ContinuousSearchData()
        cs = CausalRegressionDiscovery()
        cs.fit(X=m.get_ndarray)
        (g, s) = cs.get_result()
        edges = [(p, c) for c, y in g.parents.items() for p in y]
        logger.info(f"edge num={len(edges)}")
        self.assertGreaterEqual(len(edges), 10)
