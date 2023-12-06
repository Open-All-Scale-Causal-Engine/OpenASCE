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

from abc import ABCMeta, abstractmethod
from math import log
from typing import Tuple

import numpy as np
import pandas as pd

from .reflect_utils import new_instance


epsilon = np.finfo(np.float32).eps

def sigmoid(x):
    if isinstance(x, (float, int)):
        return 1 / (1 + np.exp(-x))
    elif isinstance(x, (np.ndarray, )):
        return np.apply_along_axis(lambda x: 1 / (1 + np.exp(-np.float32(x))), 0, x)
    else:
        raise ValueError(f'type {type(x)} not supported!')


class Loss(metaclass=ABCMeta):
    """Abstract base class for loss functions."""

    def __init__(self, **kwargs):
        self._name = kwargs.get('name', self.__class__.__name__)
        self.classification = True

    @staticmethod
    def new_instance(conf):
        """
        Create a new instance of the loss function.

        Arguments:
            conf: Configuration.

        Returns:
            An instance of the loss function.

        """
        conf = conf.get('tree', conf)
        loss_cls = conf.get('loss_cls', None)
        return new_instance(loss_cls)

    @abstractmethod
    def loss(self, target, prediction, *args):
        """
        Calculate the loss.

        Arguments:
            target: Target values.
            prediction: Predicted values.
            args: Additional arguments.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            The loss value.

        """
        raise NotImplementedError


class GradLoss(Loss):
    """Abstract base class for gradient-based loss functions."""

    @abstractmethod
    def gradients(self, target, prediction) -> Tuple:
        """
        Calculate the gradients and hessians.

        Arguments:
            target: Target values.
            prediction: Predicted values.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            Tuple containing the gradients and hessians.

        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, target, prediction):
        """
        Calculate the gradient of the loss.

        Arguments:
            target: Target values.
            prediction: Predicted values.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            The gradient.

        """
        raise NotImplementedError

    @abstractmethod
    def hessian(self, target, prediction):
        """
        Calculate the hessian of the loss.

        Arguments:
            target: Target values.
            prediction: Predicted values.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            The hessian.

        """
        raise NotImplementedError

    @property
    def const_hess(self):
        """
        Check if the hessian is constant.

        Returns:
            True if the hessian is constant, False otherwise.

        """
        return False


class MeanSquaredError(GradLoss):

    def __init__(self, **kwargs):
        self.classification = False

    def loss(self, target, prediction, *args, **kwargs):
        """
        The mean squared loss

        Arguments:
            y: [n_instance, n_outcome]
            y_hat: [n_instance, n_outcome] or [n_outcome]

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        w = kwargs.get('weight')
        w = (np.expand_dims(w, 1) if w is not None else 1)
        return np.sum(w * (target - prediction)**2, axis=0)

    def gradients(self, target, prediction, **kwargs) -> Tuple:
        if isinstance(prediction, (int, float)):
            pred = np.full_like(target, prediction, target.dtype)
        else:
            pred = prediction
        # gradient
        gs = self.gradient(target, pred, **kwargs)
        hs = self.hessian(target, pred, **kwargs)

        return gs, hs

    def gradient(self, target, prediction, **kwargs):
        w = kwargs.get('weight')
        w = (np.expand_dims(w, 1) if w is not None else 1)
        return (prediction - target) * w * 2

    def hessian(self, target, prediction, **kwargs):
        w = kwargs.get('weight')
        w = (np.expand_dims(w, 1) if w is not None else 1)
        if np.isscalar(prediction):
            return np.full_like(target, 2, target.dtype) * w
        return np.full_like(prediction, 2, prediction.dtype) * w

    @property
    def const_hess(self):
        return True


class BinaryCrossEntropy(GradLoss):

    def __init__(self, **kwargs):
        self.classification = True

    def loss(self, target, prediction, logit=True):
        """
        Calculate the cross entropy

        Arguments:
            target: ground-truth label
            prediction: prediction of logits
            logit (bool, optional): [description]. Defaults to True.
        """
        if logit:
            prob = sigmoid(prediction)
            ce = 0 - target * (prob + epsilon).apply(log) - (1 - target) * (1 - prob + epsilon).apply(log)
        else:
            ce = 0 - target * np.log(prediction + epsilon) - (1 - target) * np.log(1 - prediction + epsilon)
        return ce

    def gradients(self, target, logit, treatment):
        """
        Calculate gradient and hessian

        Arguments:
            target (DataFrame): [description]
            prediction (DataFrame): [description]
            treatment (DataFrame): [description]

        Returns:
            Union[Tuple, None]: [description]
        """
        if isinstance(logit, (int, float)):
            logit = pd.DataFrame(logit, index=target.index, columns=target.columns)
        probability = sigmoid(logit)
        # gradient
        gs = self.gradient(target, probability)
        hs = self.hessian(target, probability)

        return gs, hs

    def gradient(self, target, prediction):
        """
        Compute the gradient: gradient = prediction - target, where prediction is the positive probability.

        Arguments:
            target: The target values.
            prediction: The predicted probabilities.

        Returns:
            ndarray: The computed gradients.

        """
        return prediction - target

    def hessian(self, target, prediction):
        """
        Compute the hessian: hessian = prediction * (1 - prediction).

        Arguments:
            target: The target values.
            prediction: The predicted probabilities.

        Returns:
            ndarray: The computed hessians.

        """
        return (prediction * (1 - prediction))[target.notna()]

    @property
    def const_hess(self):
        return False
