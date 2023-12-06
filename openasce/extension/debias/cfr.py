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

from openasce.extension.debias.common.utils import DNNModel, mmd_rbf
from openasce.extension.debias_model import CausalDebiasModel


class CFRModel(CausalDebiasModel):
    """Building a CFR model.

    Model: CFR (CounterFactual Regression).

    Paper: Estimating individual treatment effect: generalization bounds and algorithms.

    Link: http://proceedings.mlr.press/v70/shalit17a/shalit17a.pdf.

    Author: Uri Shalit, Fredrik D. Johansson and David Sontag.
    """

    def __init__(self, params: typing.Dict) -> None:
        """Initialize.

        Args:
            params: parameter dict.
        """
        super().__init__()
        # initialize params.
        self.hidden_units = params.get("hidden_units", [64, 16, 1])
        self.hidden_units_emb = params.get("hidden_units_emb", [128, 64])
        self.act_fn = params.get("act_fn", "relu")
        self.l2_reg = params.get("l2_reg", 0.001)
        self.dropout_rate = params.get("dropout_rate", 0)
        self.use_bn = params.get("use_bn", False)
        self.apply_final_act = params.get("apply_final_act", False)
        self.apply_final_act_emb = params.get("apply_final_act_emb", True)
        self.lr = params.get("lr", 0.0001)
        self.proportion = params.get("proportion", 0.9)
        self.w = params.get("w", 0.1)
        # define model.
        self.model_emb = DNNModel(
            hidden_units=self.hidden_units_emb,
            act_fn=self.act_fn,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=self.apply_final_act_emb,
        )
        self.control_net = DNNModel(
            hidden_units=self.hidden_units,
            act_fn=self.act_fn,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=self.apply_final_act,
        )
        self.treatment_nets = DNNModel(
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
            self.model_emb.trainable_variables
            + self.control_net.trainable_variables
            + self.treatment_nets.trainable_variables
        )
        return variables

    def forward(
        self, x: tf.Tensor, c: typing.Dict[str, tf.Tensor], training: bool
    ) -> typing.Dict[str, tf.Tensor]:
        feature, treatment = c.get("feature"), c.get("treatment")
        features_emb = self.model_emb(feature, training=training)

        logit_control = self.control_net(features_emb, training=training)
        logit_treatment = self.treatment_nets(features_emb, training=training)
        effect = logit_treatment - logit_control

        predictions = {
            "effect": effect,
            "logit_treatment": logit_treatment,
            "logit_control": logit_control,
            "treatment": treatment,
            "features_emb": features_emb,
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
            c: the original input dict, here, {'feature': np.ndarray, 'treatment': np.ndarray}.
                feature: train feature.
                treatment:  binary treatment or multiple discrete treatment.
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
                predictions["label"] = y
            return predictions

    def loss(self, predictions: typing.Dict, labels: tf.Tensor):
        """Compute scalar loss tensors with respect to provided labels.

        Args:
            predictions: a dictionary holding predicted tensors.
            labels: label tensor.

        Returns:
            A scalar loss or A dictionary mapping strings (loss names) to scalar loss.
        """
        logit_treatment = predictions["logit_treatment"]
        logit_control = predictions["logit_control"]
        treatment = predictions["treatment"]
        features_emb = predictions["features_emb"]
        loss_treat = tf.compat.v1.losses.sigmoid_cross_entropy(
            labels, logit_treatment, weights=treatment
        )
        loss_control = tf.compat.v1.losses.sigmoid_cross_entropy(
            labels, logit_control, weights=1 - treatment
        )
        ipm_loss = mmd_rbf(features_emb, treatment, self.proportion, 2.0)
        loss = loss_treat + loss_control + self.w * ipm_loss
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
