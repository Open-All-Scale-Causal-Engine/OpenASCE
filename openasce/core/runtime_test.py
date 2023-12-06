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

from openasce.core.runtime import Runtime
from openasce.utils.logger import logger


class MockRuntime(Runtime):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        dic = {"a": 1, "b": 2}
        data = np.arange(1000).reshape((50, 20))
        results = self.launch(num=2, param=dic, dataset=data)
        logger.info(f"results={results}")

    def todo(self, idx, total_num, param, dataset):
        logger.info(
            f"id={idx}, total_num={total_num}, param={param}, dataset={dataset}"
        )
        data = dataset * (idx + 2)
        result = {k: 2 * v for k, v in param.items()}
        result["id"] = idx
        result["data"] = data
        return result


class TestRuntime(TestCase):
    def setUp(self) -> None:
        pass

    def test_runtime_execution(self) -> None:
        r = MockRuntime()
        r.fit()
