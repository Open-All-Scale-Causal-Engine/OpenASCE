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
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from ihdp_data import get_ihdp_data
from sklearn.linear_model import LassoCV

from openasce.inference.learner import DML
from openasce.utils.logger import logger

if __name__ == "__main__":
    train_data, test_data = get_ihdp_data()
    learner = DML(
        model_y=WeightedLassoCVWrapper(),
        model_t=WeightedLassoCVWrapper(),
        model_final=LassoCV(cv=3),
        categories=[0, 1],
    )
    learner.fit(
        X=train_data[train_data.columns[5:]].to_numpy().astype(np.float32),
        Y=train_data["y_factual"],
        T=train_data["treatment"],
    )
    learner.estimate(X=test_data[train_data.columns[5:]].to_numpy().astype(np.float32))
    avg = np.average(learner.get_result())
    logger.info(f"dml result: {avg}")
