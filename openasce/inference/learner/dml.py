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


from typing import Iterable, Any, NoReturn

import numpy as np
from econml.dml import DML as _DML
from econml.dml._rlearner import _ModelNuisance
from econml.utilities import filter_none_kwargs

from openasce.inference.inference_model import InferenceModel


class DML(_DML, InferenceModel):
    def fit(
        self,
        *,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray],
        T: Iterable[np.ndarray],
        **kwargs
    ):
        """Feed the sample data and train the model used to effect on the samples.

        Arguments:
            X: Features of the samples.
            Y: Outcomes of the samples.
            T: Treatments of the samples.

        Returns:

        """

        def _nuisance_fit(
            _self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None
        ):
            assert Z is None, "Cannot accept instrument!"
            param = {
                "X": X,
                "W": W,
                "T": T,
                "Y": Y,
                "model_t": _self._model_t,
                "model_y": _self._model_y,
                "sample_weight": sample_weight,
                "groups": groups,
            }
            results = self.launch(num=2, param=param, dataset=None)
            for r in results:
                if "model_t" in r:
                    _self._model_t = r["model_t"]
                elif "model_y" in r:
                    _self._model_y = r["model_y"]
            return _self

        _ModelNuisance.fit = _nuisance_fit
        super().fit(Y, T, X=X, **kwargs)

    def todo(self, idx: int, total_num: int, param: Any, dataset: Iterable) -> Any:
        model_t = param.pop("model_t")
        model_y = param.pop("model_y")
        X, Y, T, W = param["X"], param["Y"], param["T"], param["W"]
        sample_weight, groups = param["sample_weight"], param["groups"]
        result = {"idx": idx}
        if idx == 0:
            model_t.fit(
                X,
                W,
                T,
                **filter_none_kwargs(sample_weight=sample_weight, groups=groups)
            )
            result["model_t"] = model_t
        elif idx == 1:
            model_y.fit(
                X,
                W,
                Y,
                **filter_none_kwargs(sample_weight=sample_weight, groups=groups)
            )
            result["model_y"] = model_y
        return result

    def estimate(self, *, X: Iterable[np.ndarray]) -> NoReturn:
        """Feed the sample data and estimate the effect on the samples

        Arguments:
            X: Features of the samples.

        Returns:

        """
        self._estimate_result = self.const_marginal_effect(X)
