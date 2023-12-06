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

import typing

import numpy as np
import tensorflow as tf

from openasce.extension.debias.common.utils import DNNModel
from openasce.extension.debias_model import CausalDebiasModel


class IPWDebiasModel(CausalDebiasModel):
    """Building a IPW model.

    Model: IPW (Inverse Probability Weighting).

    Paper: Inverse probability weighted estimation for general missing data problems.

    Link: https://www.econstor.eu/bitstream/10419/79298/1/386079048.pdf.

    Author: Jeffrey M. Wooldridge.
    """

    def __init__(self, params: typing.Dict) -> None:
        """Initialize.

        Args:
            params: parameter dict.
        """
        super().__init__()
        # initialize params.
        self.hidden_units = params.get("hidden_units", [64, 16, 1])
        self.act_fn = params.get("act_fn", "relu")
        self.l2_reg = params.get("l2_reg", 0.001)
        self.dropout_rate = params.get("dropout_rate", 0)
        self.use_bn = params.get("use_bn", False)
        self.apply_final_act = params.get("apply_final_act", False)
        self.lr = params.get("lr", 0.0001)
        self.a_min = params.get("a_min", 0.01)
        self.a_max = params.get("a_max", 1.0)
        # define model.
        self.model = DNNModel(
            hidden_units=self.hidden_units,
            act_fn=self.act_fn,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=self.apply_final_act,
        )
        self.optimizer = self.get_optimizer()

    @property
    def trainable_variables(self):
        variables = self.model.trainable_variables
        return variables

    def forward(
        self, x: tf.Tensor, c: typing.Dict[str, tf.Tensor], training: bool
    ) -> typing.Dict[str, tf.Tensor]:
        feature, weight = c.get("feature"), c.get("weight")
        logits = self.model(feature, training=training)

        predictions = {
            "logits": logits,
            "weight": weight,
        }
        return predictions

    def _call(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        c: typing.Dict[str, np.ndarray],
        training: bool
    ) -> typing.Union[None, typing.Dict[str, np.ndarray]]:
        """Building a callable function.
            fit and predict are the base class interface methods to be called by outside users, which should not be overloaded.
            _call is used to implement the logic of the algorithm after it has been overloaded.

        Args:
            x: the original input feature.
            y: the original input label.
            c: the original input dict, here, {'feature': np.ndarray, 'weight': np.ndarray}.
                feature: train feature.
                weight: indicates the exposure proportion of item perspective.
            training: bool, identify the status.
        Returns:
            A callable function,
                for training, return loss, optimizer, and model;
                for inference, return the prediction dict.
        """

        def grad(x, c, training, labels):
            with tf.GradientTape() as tape:
                predictions = self.forward(x, c, training)
                loss_value = self.loss(predictions, labels)
            return loss_value, tape.gradient(loss_value, self.trainable_variables)

        if training:
            # train procedure
            # calculate loss, gradient, optimizer updates model, etc.
            # The framework doesn't care about return values.
            loss_value, grads = grad(x, c, training, y)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        else:
            # inference procedure.
            # calculate the prediction and return with a dict.
            predictions = self.forward(x, c, training)
            if y is not None:
                predictions["labels"] = y
            return predictions

    def loss(self, predictions: typing.Dict, labels: tf.Tensor):
        """Compute scalar loss tensors with respect to provided labels.

        Args:
            predictions: a dictionary holding predicted tensors.
            labels: label tensor dict.

        Returns:
            A scalar loss or A dictionary mapping strings (loss names) to scalar loss.
        """
        logits = predictions["logits"]
        weight = predictions["weight"]

        # a_min, a_max: hyper-parameters.
        a_min = self.a_min
        a_max = self.a_max

        weights = tf.clip_by_value(1 / (weight), a_min, a_max)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits, weights)
        return loss

    def get_optimizer(self):
        """Build the optimizer.

        Args:

        Returns:
            An optimizer.
        """
        # lr hyper-parameters.
        lr = self.lr
        return tf.keras.optimizers.Adam(lr=lr)
