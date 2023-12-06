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

from functools import wraps
from typing import Any, Iterable

import numpy as np

from openasce.core.runtime import Runtime
from openasce.utils.logger import logger


class InferenceModel(Runtime):
    """Inference Class

    Base class of the causal inference

    Attributes:

    """

    CONDITION_DICT_NAME = "condition"
    TREATMENT_VALUE = "treatment_value"
    LABEL_VALUE = "label_value"

    def __init__(self) -> None:
        super().__init__()

    @property
    def data(self):
        """Return the sample data"""
        raise NotImplementedError(f"Not implement for abstract class")

    def fit(
        self,
        *,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray],
        T: Iterable[np.ndarray],
        **kwargs,
    ) -> None:
        """Feed the sample data and train the model used to effect on the samples.

        Arguments:
            X: Features of the samples.
            Y: Outcomes of the samples.
            T: Treatments of the samples.

        Returns:
            None
        """
        pass

    def estimate(
        self,
        *,
        X: Iterable[np.ndarray],
        T: Iterable[np.ndarray],
        **kwargs,
    ) -> None:
        """Feed the sample data and estimate the effect on the samples

        Arguments:
            X: Features of the samples.
            T: Treatments of the samples.

        Returns:
            None
        """
        pass

    def get_result(self) -> Any:
        """Get the estimated result

        The sub-class should implement this routine and runtime invokes it.

        Returns:
            The estimation result.
        """
        return self._estimate_result

    def output(self, output_path: str) -> None:
        """Output the estimated result to files

        The sub-class should implement this routine and runtime invokes it.

        Arguments:
            output_path: The path of output file.

        Returns:
            None
        """
        from numpy import savetxt

        savetxt(output_path, self.get_result())
        logger.info(f"Write result to file: {output_path}")

    def _wrap_fit(m):
        @wraps(m)
        def call(self, *, X, Y, T, **kwargs):
            self._prefit(Y, T, X=X, **kwargs)
            # call the wrapped fit method
            m(self, X=X, Y=Y, T=T, **kwargs)
            self._postfit(Y, T, X=X, **kwargs)
            return self

        return call
