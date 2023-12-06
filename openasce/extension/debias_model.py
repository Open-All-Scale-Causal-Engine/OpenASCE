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

from typing import Dict, Iterable, Union, Tuple

import numpy as np

from openasce.core.runtime import Runtime
from openasce.utils.logger import logger


class CausalDebiasModel(Runtime):
    """Debias Inference Class

    Base class of the causal debias

    Attributes:

    """

    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        *,
        X: Iterable[np.ndarray] = None,
        Y: Iterable[np.ndarray] = None,
        C: Dict[str, Iterable[np.ndarray]] = None,
        Z: Iterable[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]] = None,
        num_epochs: int = 1,
        **kwargs,
    ) -> None:
        """Feed the sample data and train the model on the samples.

        Arguments:
            X: Features of the samples.
            Y: Outcomes of the samples.
            C: Other concerned columns of the samples, e.g. {'weight': Iterable[np.ndarray]}
            Z: The iterable object returning (a batch of X, a batch of Y, a batch of C) if having
            num_epochs: number of the train epoch
        Returns:
            None
        """
        if C is not None and not isinstance(C, dict):
            raise ValueError(f"C should be dict.")
        if not (X or Y or C or Z):
            raise ValueError(f"One of (X, Y, C, Z) should be set.")
        self._X = X
        self._Y = Y
        self._C = C
        self._Z = Z
        self._train_loop(num_epochs=num_epochs, **kwargs)

    def predict(
        self,
        *,
        X: Iterable[np.ndarray] = None,
        C: Dict[str, Iterable[np.ndarray]] = None,
        Z: Iterable[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]] = None,
        **kwargs,
    ) -> None:
        """Feed the sample data

        Estimate the effect on the samples, and get_result method can be used to get the result of prediction

        Arguments:
            X: Features of the samples.
            C: Other concerned columns of the samples, e.g. {'weight': Iterable[np.ndarray]}
            Z: The iterable object returning (a batch of X, a batch of Y, a batch of C) if having

        Returns:
            None
        """
        if C is not None and not isinstance(C, dict):
            raise ValueError(f"C should be dict.")
        if not (X or C or Z):
            raise ValueError(f"One of (X, C, Z) should be set.")
        self._X = X
        self._C = C
        self._Y = None
        self._Z = Z
        self._result = self._predict_loop(**kwargs)

    def get_result(self):
        """Get the predict result

        Arguments:

        Returns:
            predict result
        """
        return self._result

    def _call(
        self, *, x: np.ndarray, y: np.ndarray, c: Dict[str, np.ndarray], training: bool
    ) -> Union[None, Dict[str, np.ndarray]]:
        """
        The derived class should override this method to train the model using loss_object and optimizer or predict on the samples.

        Arguments:
            x: one batch of features
            y: one batch of labels
            c: one batch for each concerned columns of the samples, e.g. {'weight': Iterable[np.ndarray]}
            training: True means training and False for predict
        Returns:
            None for training and Dict for predict
        Raise:
            StopIteration: The process can be finished
        """
        raise NotImplementedError(f"Not implementation for _call method")

    def _train_loop(self, *, num_epochs, **kwargs):
        """main loop for train"""

        curr_epoch = 0
        while curr_epoch < num_epochs:
            for z in self._generator():
                self._call(
                    x=z[0] if len(z) > 0 else None,
                    y=z[1] if len(z) > 1 else None,
                    c=z[2] if len(z) > 2 else None,
                    training=True,
                )
            logger.info(f"Finish epoch {curr_epoch}.")
            curr_epoch += 1

    def _predict_loop(self):
        """main loop for prediction"""
        f_result = {}
        for z in self._generator():
            result = self._call(
                x=z[0] if len(z) > 0 else None,
                y=z[1] if len(z) > 1 else None,
                c=z[2] if len(z) > 2 else None,
                training=False,
            )
            for k, v in result.items():
                f_result[k] = np.hstack([f_result[k], v]) if k in f_result else v
        return f_result

    def _generator(self, **kwargs):
        """main loop"""

        def none_generator():
            while True:
                yield None

        if self._Z:
            iz = iter(self._Z)
        else:
            ix = iter(self._X) if self._X else none_generator()
            iy = iter(self._Y) if self._Y else none_generator()
            ics = (
                [(i[0], iter(i[1])) for i in self._C.items()]
                if self._C
                else {"nonsense_placeholder": none_generator()}
            )
            iz = map(
                lambda _: (
                    next(ix),
                    next(iy),
                    {k: next(v) for k, v in ics},
                ),
                none_generator(),
            )
        return iz
