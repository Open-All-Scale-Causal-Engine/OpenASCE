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

import numpy as np

from openasce.discovery.causal_graph import CausalGraph
from openasce.inference.graph_inference import GraphInferModel
from openasce.utils.logger import logger


class TestGraphInference(TestCase):
    def setUp(self) -> None:
        pass

    def test_graph_inference(self) -> None:
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
        logger.info(f"{gi.label_name}")
        gi.fit(X=d)
        gi.estimate(
            X=None,
            Y=None,
            condition={"x0": 0},
            treatment_value=1,
        )
        result = gi.get_result()
        logger.info(f"{result.get(gi.label_name)}")

        gi.estimate(
            X=None,
            Y=None,
            condition={
                "x0": 1,
                "x2": 1,
                "x3": 1,
                "x6": 1,
            },
            treatment_value=1,
        )
        result = gi.get_result()
        # output result for confirm
        for _ in map(
            lambda p: logger.info(f"name={p[0]}, result={p[1]}"),
            result.items(),
        ):
            pass
        r = result.get("x9")
        self.assertEqual(
            len(
                list(
                    filter(
                        lambda x: (x[0] - x[1]) > 0.0001,
                        zip(
                            r.data[:, r.score_column_index],
                            np.array([0.75512414, 0.24487582]),
                        ),
                    )
                )
            ),
            0,
        )
        r = result.get(m.label_name)
        self.assertEqual(
            len(
                list(
                    filter(
                        lambda x: (x[0] - x[1]) > 0.0001,
                        zip(
                            r.data[:, r.score_column_index],
                            np.array([0.76321607, 0.23678391]),
                        ),
                    )
                )
            ),
            0,
        )
