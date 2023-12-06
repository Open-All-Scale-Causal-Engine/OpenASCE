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

import os

import numpy as np
from ihdp_data import get_ihdp_data
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

from openasce.inference.learner import SLearner, TLearner, XLearner
from openasce.utils.logger import logger

train_data, test_data = get_ihdp_data()


def tlearner():
    learner = TLearner(
        models=[GradientBoostingRegressor(), GradientBoostingRegressor()]
    )
    learner.fit(
        X=train_data[train_data.columns[5:]].to_numpy().astype(np.float32),
        Y=train_data["y_factual"],
        T=train_data["treatment"],
    )
    learner.estimate(X=test_data[train_data.columns[5:]].to_numpy().astype(np.float32))
    avg = np.average(learner.get_result())
    logger.info(f"tlearner result: {avg}")
    learner.output(output_path="tmp_result.txt")
    os.remove("tmp_result.txt")


def slearner():
    learner = SLearner(models=[GradientBoostingRegressor()], categories=[0, 1])
    learner.fit(
        X=train_data[train_data.columns[5:]].to_numpy().astype(np.float32),
        Y=train_data["y_factual"],
        T=train_data["treatment"],
    )
    learner.estimate(X=test_data[train_data.columns[5:]].to_numpy().astype(np.float32))
    avg = np.average(learner.get_result())
    logger.info(f"slearner result: {avg}")


def xlearner():
    learner = XLearner(
        models=[GradientBoostingRegressor(), GradientBoostingRegressor()],
        cate_models=None,
        propensity_model=RandomForestClassifier(),
        categories=[0, 1],
    )
    learner.fit(
        X=train_data[train_data.columns[5:]].to_numpy().astype(np.float32),
        Y=train_data["y_factual"],
        T=train_data["treatment"],
    )
    learner.estimate(X=test_data[train_data.columns[5:]].to_numpy().astype(np.float32))
    avg = np.average(learner.get_result())
    logger.info(f"xlearner result: {avg}")


if __name__ == "__main__":
    # Uncomment as needed to run the learner.
    tlearner()
    # slearner()
    # xlearner()
