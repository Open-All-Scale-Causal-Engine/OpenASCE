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
from tensorflow.python.keras.layers import Concatenate

from openasce.extension.debias.common.utils import DNNModel, FMLayer
from openasce.extension.debias_model import CausalDebiasModel


class DMBRDebiasModel(CausalDebiasModel):
    """Building a DMBR model.

    Model: DMBR (De-Matching Bias Recommendation).

    Paper: Alleviating Matching Bias in Marketing Recommendations.

    Link: https://dl.acm.org/doi/abs/10.1145/3539618.3591854.

    Author: Junpeng Fang, Qing Cui, Gongduo Zhang, Caizhi Tang, Lihong Gu, Longfei Li, Jinjie Gu, Jun Zhou, Fei Wu.

    """

    def __init__(self, params: typing.Dict) -> None:
        """Initialize.

        Args:
            params: parameter dict.
        """
        super().__init__()
        # initialize params.
        self.hidden_units = params.get("hidden_units", [64, 16, 1])
        self.hidden_units_emb = params.get("hidden_units_emb", [64, 16])
        self.act_fn = params.get("act_fn", "relu")
        self.l2_reg = params.get("l2_reg", 0.001)
        self.dropout_rate = params.get("dropout_rate", 0)
        self.use_bn = params.get("use_bn", False)
        self.apply_final_act = params.get("apply_final_act", False)
        self.apply_final_act_emb = params.get("apply_final_act_emb", True)
        self.lr = params.get("lr", 0.0001)

        # define model.
        self.model_emb_1 = DNNModel(
            hidden_units=self.hidden_units_emb,
            act_fn=self.act_fn,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=self.apply_final_act_emb,
        )
        self.model_emb_2 = DNNModel(
            hidden_units=self.hidden_units_emb,
            act_fn=self.act_fn,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=self.apply_final_act_emb,
        )
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
        variables = (
            self.model_emb_1.trainable_variables
            + self.model_emb_2.trainable_variables
            + self.model.trainable_variables
        )
        return variables

    def forward(
        self, x: tf.Tensor, c: typing.Dict[str, tf.Tensor], training: bool
    ) -> typing.Dict[str, tf.Tensor]:
        feature, confounder = c.get("feature"), c.get("confounder")
        confounder_emb = self.model_emb_1(confounder, training=training)
        feature_emb = self.model_emb_2(feature, training=training)
        confounder_mean = tf.compat.v1.reduce_mean(confounder_emb, 0, keep_dims=True)
        confounder_mean_overall = tf.tile(confounder_mean, [tf.shape(confounder)[0], 1])
        inputs = [feature_emb, confounder_mean_overall]
        inputs_c = Concatenate(axis=-1)(inputs)
        union_input = tf.reshape(inputs_c, [-1, len(inputs), self.hidden_units_emb[-1]])
        confounder_union = FMLayer()(inputs=union_input)
        logits = self.model(
            tf.concat([feature, confounder_union], axis=1), training=training
        )

        predictions = {
            "logits": logits,
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
            c: the original input dict, here, {'feature': np.ndarray, 'confounder': np.ndarray}.
                feature: train feature.
                confounder: confounder feature.
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
