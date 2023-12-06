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

from openasce.attribution.attribution_model import Attribution
from openasce.discovery.causal_graph import CausalGraph
from openasce.inference.graph_inference import GraphInferModel
from openasce.utils.logger import logger


class TestAttribution(TestCase):
    def setUp(self) -> None:
        pass

    def test_attribution(self) -> None:
        from tests.datasets.mock_graph_infer_data import MockCausalInferData

        m = MockCausalInferData()
        g = CausalGraph(names=m.all_column_names)
        add_edge = lambda g: lambda x: g.add_edge(x[0], x[1])
        for _ in map(add_edge(g), m.parents):
            pass
        d = m.get_all_data()
        gi = GraphInferModel()
        gi.graph = g
        gi.treatment_name = m.treatment_name
        gi.label_name = m.label_name

        attr = Attribution(threshold=0.1, max_step=2)
        attr.inferencer = gi
        attr.attribute(X=d, treatment_value=1, label_value=1)
        result = attr.get_result()
        self.assertGreater(len(result), 9)
