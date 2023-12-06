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


from typing import List, Union, Iterable, NoReturn, Any

import numpy as np
from econml.metalearners import (
    TLearner as _TLearner,
    SLearner as _SLearner,
    XLearner as _XLearner,
)
from econml.utilities import check_inputs, check_models, inverse_onehot
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from openasce.inference.inference_model import InferenceModel


class TLearner(_TLearner, InferenceModel):
    def __init__(self, *, models: List, categories: Union[str, list] = "auto") -> None:
        """Initialize TLearner

        Args:
            models (List): List of outcome estimators for both control units and treatment units, all models predictions result must contain `prediction_key`.
            categories (List[Union[int, float]]): List of treatments values(like [0,1], 0 is for control). The first category will be treated as the control treatment.
        """
        super().__init__(models=models, categories=categories)

    @InferenceModel._wrap_fit
    def fit(
        self,
        *,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray],
        T: Iterable[np.ndarray],
        **kwargs
    ) -> NoReturn:
        """Feed the sample data and train the model used to effect on the samples.

        Arguments:
            X: Features of the samples.
            Y: Outcomes of the samples.
            T: Treatments of the samples.

        Returns:

        """
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False)
        categories = self.categories
        if categories != "auto":
            categories = [
                categories
            ]  # OneHotEncoder expects a 2D array with features per column
        self.transformer = OneHotEncoder(
            categories=categories, sparse=False, drop="first"
        )
        T = self.transformer.fit_transform(T.reshape(-1, 1))
        self._d_t = T.shape[1:]
        T = inverse_onehot(T)
        self.models = check_models(self.models, self._d_t[0] + 1)
        results = self.launch(
            num=self._d_t[0] + 1, param={"X": X, "Y": Y, "T": T}, dataset=None
        )
        for r in results:
            self.models[r["idx"]] = r["model"]

    def todo(self, idx: int, total_num: int, param: Any, dataset: Iterable) -> Any:
        model = self.models[idx]
        model.fit(param["X"][param["T"] == idx], param["Y"][param["T"] == idx])
        result = {"idx": idx, "model": model}
        return result

    def estimate(self, *, X: Iterable[np.ndarray]) -> NoReturn:
        """Feed the sample data and estimate the effect on the samples

        Arguments:
            X: Features of the samples.

        Returns:

        """
        self._estimate_result = self.const_marginal_effect(X)


class SLearner(_SLearner, InferenceModel):
    def __init__(self, *, models: List, categories: Union[str, list] = "auto") -> None:
        """Initialize SLearner

        Args:
            models : Outcome estimators for all units, only need one model.
            categories (List[Union[int, float]]): List of treatments values(like [0,1], 0 is for control). The first category will be treated as the control treatment.
        """
        assert len(models) == 1, "SLearner only support one model"
        super().__init__(overall_model=models[0], categories=categories)

    @InferenceModel._wrap_fit
    def fit(
        self,
        *,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray],
        T: Iterable[np.ndarray],
        **kwargs
    ) -> NoReturn:
        """Feed the sample data and train the model used to effect on the samples.

        Arguments:
            X: Features of the samples.
            Y: Outcomes of the samples.
            T: Treatments of the samples.

        Returns:

        """
        if X is None:
            X = np.zeros((Y.shape[0], 1))
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False)
        categories = self.categories
        if categories != "auto":
            categories = [
                categories
            ]  # OneHotEncoder expects a 2D array with features per column
        self.transformer = OneHotEncoder(
            categories=categories, sparse=False, drop="first"
        )
        T = self.transformer.fit_transform(T.reshape(-1, 1))
        self._d_t = (T.shape[1],)
        feat_arr = np.concatenate((X, 1 - np.sum(T, axis=1).reshape(-1, 1), T), axis=1)
        results = self.launch(num=1, param={"feat_arr": feat_arr, "Y": Y}, dataset=None)
        self.overall_model = results[0]

    def todo(self, idx: int, total_num: int, param: Any, dataset: Iterable) -> Any:
        self.overall_model.fit(param["feat_arr"], param["Y"])
        return self.overall_model

    def estimate(self, *, X: Iterable[np.ndarray]) -> NoReturn:
        """Feed the sample data and estimate the effect on the samples

        Arguments:
            X: Features of the samples.

        Returns:

        """
        self._estimate_result = self.const_marginal_effect(X)


class XLearner(_XLearner, InferenceModel):
    def __init__(
        self,
        *,
        models: List,
        cate_models: List = None,
        propensity_model=LogisticRegression(),
        categories: Union[str, list] = "auto"
    ) -> None:
        """Initialize XLearner

        Args:
            models (List): outcome estimators for both control units and treatment units, all models predictions result must contain `prediction_key`.
            cate_models (List): estimator for pseudo-treatment effects on control and treatments, all models predictions result must contain `prediction_key`.
            propensity_model : estimator for the propensity function, `propensity_model` predictions result must contain `prediction_key`.
            categories (List[Union[int, float]]): list of treatments values(like [0,1], 0 is for control). The first category will be treated as the control treatment.
        """
        super().__init__(
            models=models,
            cate_models=cate_models,
            propensity_model=propensity_model,
            categories=categories,
        )

    @InferenceModel._wrap_fit
    def fit(
        self,
        *,
        X: Iterable[np.ndarray],
        Y: Iterable[np.ndarray],
        T: Iterable[np.ndarray],
        **kwargs
    ) -> NoReturn:
        """Feed the sample data and train the model used to effect on the samples.

        Arguments:
            X: Features of the samples.
            Y: Outcomes of the samples.
            T: Treatments of the samples.

        Returns:

        """
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False)
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.flatten()
        categories = self.categories
        if categories != "auto":
            categories = [categories]
        self.transformer = OneHotEncoder(
            categories=categories, sparse=False, drop="first"
        )
        T = self.transformer.fit_transform(T.reshape(-1, 1))
        self._d_t = T.shape[1:]
        T = inverse_onehot(T)
        self.models = check_models(self.models, self._d_t[0] + 1)
        if self.cate_models is None:
            self.cate_models = [clone(model, safe=False) for model in self.models]
        else:
            self.cate_models = check_models(self.cate_models, self._d_t[0] + 1)
        self.propensity_models = [None] * self._d_t[0]
        self.cate_treated_models = [None] * self._d_t[0]
        self.cate_controls_models = [None] * self._d_t[0]

        model_results = self.launch(
            num=self._d_t[0] + 1,
            param={"stage": 0, "X": X, "Y": Y, "T": T},
            dataset=None,
        )
        for mr in model_results:
            self.models[mr["idx"]] = mr["model"]
        results = self.launch(
            num=self._d_t[0], param={"stage": 1, "X": X, "Y": Y, "T": T}, dataset=None
        )
        for r in results:
            self.propensity_models[r["idx"]] = r["propensity_model"]
            self.cate_treated_models[r["idx"]] = r["cate_treated_model"]
            self.cate_controls_models[r["idx"]] = r["cate_controls_model"]

    def todo(self, idx: int, total_num: int, param: Any, dataset: Iterable) -> Any:
        if param["stage"] == 0:
            self.models[idx].fit(
                param["X"][param["T"] == idx], param["Y"][param["T"] == idx]
            )
            result = {"idx": idx, "model": self.models[idx]}
        elif param["stage"] == 1:
            X, Y, T = param["X"], param["Y"], param["T"]
            cate_treated_model = clone(self.cate_models[idx + 1], safe=False)
            cate_controls_model = clone(self.cate_models[0], safe=False)
            propensity_model = clone(self.propensity_model, safe=False)
            imputed_effect_on_controls = (
                self.models[idx + 1].predict(X[T == 0]) - Y[T == 0]
            )
            imputed_effect_on_treated = Y[T == idx + 1] - self.models[0].predict(
                X[T == idx + 1]
            )
            cate_controls_model.fit(X[T == 0], imputed_effect_on_controls)
            cate_treated_model.fit(X[T == idx + 1], imputed_effect_on_treated)
            X_concat = np.concatenate((X[T == 0], X[T == idx + 1]), axis=0)
            T_concat = np.concatenate((T[T == 0], T[T == idx + 1]), axis=0)
            propensity_model.fit(X_concat, T_concat)
            result = {
                "idx": idx,
                "cate_controls_model": cate_controls_model,
                "cate_treated_model": cate_treated_model,
                "propensity_model": propensity_model,
            }
        else:
            raise NotImplementedError()

        return result

    def estimate(self, *, X: Iterable[np.ndarray]) -> NoReturn:
        """Feed the sample data and estimate the effect on the samples

        Arguments:
            X: Features of the samples.

        Returns:

        """
        self._estimate_result = self.const_marginal_effect(X)
