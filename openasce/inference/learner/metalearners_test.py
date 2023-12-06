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
from unittest import TestCase

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

from openasce.inference.learner.metalearners import SLearner, TLearner, XLearner
from tests.datasets.ihdp_data import get_ihdp_data
from openasce.utils.logger import logger


class TestMetaLearners(TestCase):
    def setUp(self) -> None:
        self.train_data, self.test_data = get_ihdp_data()
        np.random.seed(12)
        return super().setUp()

    def test_tlearner(self):
        learner = TLearner(
            models=[
                GradientBoostingRegressor(random_state=12),
                GradientBoostingRegressor(random_state=12),
            ]
        )
        learner.fit(
            X=self.train_data[self.train_data.columns[5:]]
            .to_numpy()
            .astype(np.float32),
            Y=self.train_data["y_factual"],
            T=self.train_data["treatment"],
        )
        learner.estimate(
            X=self.test_data[self.train_data.columns[5:]].to_numpy().astype(np.float32)
        )
        avg = np.average(learner.get_result())
        logger.info(f"tlearner result: {avg}")
        self.assertAlmostEqual(avg, 3.85, delta=0.1)
        learner.output(output_path="tmp_result.txt")
        os.remove("tmp_result.txt")

    def test_slearner(self):
        learner = SLearner(
            models=[GradientBoostingRegressor(random_state=12)], categories=[0, 1]
        )
        learner.fit(
            X=self.train_data[self.train_data.columns[5:]]
            .to_numpy()
            .astype(np.float32),
            Y=self.train_data["y_factual"],
            T=self.train_data["treatment"],
        )
        learner.estimate(
            X=self.test_data[self.train_data.columns[5:]].to_numpy().astype(np.float32)
        )
        avg = np.average(learner.get_result())
        logger.info(f"slearner result: {avg}")
        self.assertAlmostEqual(avg, 3.8, delta=0.1)

    def test_xlearner(self):
        learner = XLearner(
            models=[
                GradientBoostingRegressor(random_state=12),
                GradientBoostingRegressor(random_state=12),
            ],
            cate_models=None,
            propensity_model=RandomForestClassifier(),
            categories=[0, 1],
        )
        learner.fit(
            X=self.train_data[self.train_data.columns[5:]]
            .to_numpy()
            .astype(np.float32),
            Y=self.train_data["y_factual"],
            T=self.train_data["treatment"],
        )
        learner.estimate(
            X=self.test_data[self.train_data.columns[5:]].to_numpy().astype(np.float32)
        )
        avg = np.average(learner.get_result())
        logger.info(f"xlearner result: {avg}")
        self.assertAlmostEqual(avg, 3.9, delta=0.1)
