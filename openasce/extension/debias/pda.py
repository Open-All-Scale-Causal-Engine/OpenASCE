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


class PDADebiasModel(CausalDebiasModel):
    """Building a PDA model.

    Model: PDA (Popularity-bias Deconfounding and Adjusting).

    Link: https://arxiv.org/pdf/2105.06067.pdf.

    Author: Yang Zhang, Fuli Feng, Xiangnan He, Tianxin Wei, Chonggang Song, Guohui Ling and Yongdong Zhang.

    """

    def __init__(self, params: typing.Dict) -> None:
        """Initialize.

        Args:
            params: parameter dict.
        """
        super().__init__()
        # initialize params.
        self.hidden_units = params.get("hidden_units", [64, 16, 1])
        self.act_fn = params.get("act_fn", tf.nn.elu)
        self.l2_reg = params.get("l2_reg", 0.001)
        self.dropout_rate = params.get("dropout_rate", 0)
        self.use_bn = params.get("use_bn", False)
        self.apply_final_act = params.get("apply_final_act", False)
        self.lr = params.get("lr", 0.0001)
        self.gamma = params.get("gamma", 0.1)
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
        feature, mit = c.get("feature"), c.get("mit")
        logits_pd = self.model(feature, training=training)
        mit = tf.pow(
            tf.cast(mit, tf.float32), tf.constant(self.gamma, dtype=tf.float32)
        )
        logits_pda = logits_pd * mit

        predictions = {
            "logits_pd": logits_pd,
            "logits_pda": logits_pda,
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
            c: the original input dict, here, {'feature': np.ndarray, 'mit': np.ndarray}.
                feature: train feature.
                mit: indicte the popularity of the item.
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
            x: the original input.

        Returns:
            A scalar loss or A dictionary mapping strings (loss names) to scalar loss.
        """
        logits = predictions["logits_pd"]
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
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
