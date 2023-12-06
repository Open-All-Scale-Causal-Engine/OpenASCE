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
from econml.dr import DRLearner as _DRLearner
from econml.dr._drlearner import _ModelNuisance
from econml.utilities import filter_none_kwargs, inverse_onehot, fit_with_groups

from openasce.inference.inference_model import InferenceModel


class DRLearner(_DRLearner, InferenceModel):
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
            _self, Y, T, X=None, W=None, *, sample_weight=None, groups=None
        ):
            if Y.ndim != 1 and (Y.ndim != 2 or Y.shape[1] != 1):
                raise ValueError(
                    "The outcome matrix must be of shape ({0}, ) or ({0}, 1), "
                    "instead got {1}.".format(len(X), Y.shape)
                )
            if (X is None) and (W is None):
                raise AttributeError("At least one of X or W has to not be None!")
            if np.any(np.all(T == 0, axis=0)) or (not np.any(np.all(T == 0, axis=1))):
                raise AttributeError(
                    "Provided crossfit folds contain training splits that "
                    + "don't contain all treatments"
                )
            XW = _self._combine(X, W)
            param = {
                "X": X,
                "XW": XW,
                "T": T,
                "Y": Y,
                "model_propensity": _self._model_propensity,
                "model_regression": _self._model_regression,
                "sample_weight": sample_weight,
                "groups": groups,
            }
            results = self.launch(num=2, param=param, dataset=None)
            for r in results:
                if "model_propensity" in r:
                    _self._model_propensity = r["model_propensity"]
                elif "model_regression" in r:
                    _self._model_regression = r["model_regression"]
            return _self

        _ModelNuisance.fit = _nuisance_fit
        super().fit(Y, T, X=X, **kwargs)

    def todo(self, idx: int, total_num: int, param: Any, dataset: Iterable) -> Any:
        model_propensity = param.pop("model_propensity")
        model_regression = param.pop("model_regression")
        X, Y, T, XW = param["X"], param["Y"], param["T"], param["XW"]
        sample_weight, groups = param["sample_weight"], param["groups"]
        filtered_kwargs = filter_none_kwargs(sample_weight=sample_weight)
        result = {"idx": idx}
        if idx == 0:
            fit_with_groups(
                model_propensity,
                XW,
                inverse_onehot(T),
                groups=groups,
                **filtered_kwargs
            )
            result["model_propensity"] = model_propensity
        elif idx == 1:
            fit_with_groups(
                model_regression,
                np.hstack([XW, T]),
                Y,
                groups=groups,
                **filtered_kwargs
            )
            result["model_regression"] = model_regression

        return result

    def estimate(self, *, X: Iterable[np.ndarray]) -> NoReturn:
        """Feed the sample data and estimate the effect on the samples

        Arguments:
            X: Features of the samples.

        Returns:

        """
        self._estimate_result = self.const_marginal_effect(X)
