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

import numpy as np

from openasce.attribution import Attribution
from openasce.discovery import CausalGraph
from openasce.inference import GraphInferModel
from openasce.utils.logger import logger


def main():
    X = np.loadtxt("attribution_samples.csv", delimiter=",", dtype=np.int32)

    # The causal graph used in attribution
    g = CausalGraph(
        names=[
            CausalGraph.DEFAULT_COLUMN_NAME_PREFIX + str(i) for i in range(X.shape[1])
        ]
    )
    parents_data = np.loadtxt("causal_graph.csv", delimiter=",", dtype=np.int32)
    graph_parents = [
        [
            CausalGraph.DEFAULT_COLUMN_NAME_PREFIX + str(i[0]),
            CausalGraph.DEFAULT_COLUMN_NAME_PREFIX + str(i[1]),
        ]
        for i in parents_data
    ]
    add_edge = lambda g: lambda x: g.add_edge(x[0], x[1])
    for _ in map(add_edge(g), graph_parents):
        pass

    # The inference model used in the attribution
    gi = GraphInferModel()
    gi.graph = g
    gi.treatment_name = CausalGraph.DEFAULT_COLUMN_NAME_PREFIX + str(7)
    gi.label_name = CausalGraph.DEFAULT_COLUMN_NAME_PREFIX + str(1)

    attr = Attribution(threshold=0.1, max_step=2)
    # Set the inferencer to attribution model
    attr.inferencer = gi
    attr.attribute(X=X, treatment_value=1, label_value=1)
    result = attr.get_result()
    print(result)


if __name__ == "__main__":
    main()
