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

import concurrent.futures
from multiprocessing import cpu_count
from typing import Any, Iterable, List

from openasce.utils.logger import logger


class Runtime:
    """Runtime Class

    Provide the runtime layer to support different running environment, including the single machine or multiple machines.

    Attributes:

    """

    def __init__(self) -> None:
        super().__init__()

    def launch(
        self, *, num: int = 1, param: Any = None, dataset: Iterable = None
    ) -> List:
        """Start the job on current environment

        The function is called as the start point of one causal workload and setup the instances according to current environment. Iterable[Tuple[np.ndarray, np.ndarray]]

        Arguments:

        Returns:

        """
        # TODO: In distributed environment, launch will setup the environment and submit the job. Then the object of the class needs to be created in workers, and then execute _instance_launch method.
        # self._instance_launch()  # For now run in same process for single machine.
        ids = [i for i in range(num)]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=cpu_count()
        ) as executor:
            to_do = []
            for id in ids:
                futu = executor.submit(self._instance_launch, id, num, param, dataset)
                to_do.append(futu)
        finals = concurrent.futures.as_completed(to_do)
        fi = [f for f in finals]
        results = [f.result() for f in fi]
        return results

    def _instance_launch(
        self, idx: int, total_num: int, param: Any, dataset: Iterable
    ) -> Any:
        """Running on the instance with multiple cores

        Arguments:

        Returns:

        """
        # TODO: Prepare the worker running environment then call todo method, which should be overloaded by sub-class and implement the function.
        logger.info(f"Begin to execute todo: {idx}/{total_num}")
        result = self.todo(idx, total_num, param, dataset)
        logger.info(f"Finish execute todo: {idx}/{total_num}")
        return result

    def todo(self, id: int, total_num: int, param: Any, dataset: Iterable) -> Any:
        """Contain the function from the sub-class, and run it in workers

        The sub-class should implement this routine and runtime invokes it.

        Arguments:

        Returns:

        """
        raise NotImplementedError(f"Not implement for abstract class")
