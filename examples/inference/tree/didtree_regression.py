#    Copyright 2023 AntGroup CO., Ltd.
#
#    Licensed under the Apache License, Version 2.0 (the 'License');
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an 'AS IS' BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os

import numpy as np
import pandas as pd

from openasce.inference.tree import DifferenceInDifferencesRegressionTree
from openasce.inference.tree.utils import to_row_major


def read_data(paths):
    X = pd.read_csv(paths[0], header=0, index_col=0)
    Y = pd.read_csv(paths[1], header=[0, 1], index_col=0)
    features = X[[f"conf_{i}" for i in range(10)] + [f"cov_{i}" for i in range(10)]]
    treatment = X[["treatment"]]
    targets = Y[[("y", i) for i in Y["y"].columns]]
    tau = Y[("eff", "1")]
    return (
        to_row_major(features),
        to_row_major(targets),
        to_row_major(treatment, np.int32),
        to_row_major(tau),
    )


if __name__ == "__main__":
    data_path = "../data/simulation/"
    features, targets, treatment, tau = read_data(
        [os.path.join(data_path, "X.csv"), os.path.join(data_path, "Y.csv")]
    )
    cv = np.load(os.path.join(data_path, "tr_te.npy"))
    pehes = []
    for i in range(10):
        tr_idx, te_idx = cv[i, 0], cv[i, 1]
        m = DifferenceInDifferencesRegressionTree(
            n_period=8, treat_dt=7, coeff=0.5, parallel_l2=10
        )
        m.fit(features[tr_idx], targets[tr_idx], treatment[tr_idx, 0])
        # leaf_ids = m.predict(features[te_idx], 'leaf_id')
        # print(leaf_ids)
        tau_hat = m.effect(features[te_idx])
        print(f"Error_MAE:{((abs(tau[te_idx]-tau_hat[:, 0])).mean()):.3f}")
        pehes.append((abs(tau[te_idx] - tau_hat[:, 0])).mean())
    pehes = np.asarray(pehes)
    print(f"{pehes.mean():.2f}, {pehes.std():.2f}")
