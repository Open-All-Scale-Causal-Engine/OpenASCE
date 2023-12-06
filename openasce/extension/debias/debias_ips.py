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

from typing import Dict, List, Union

import numpy as np
import tensorflow as tf

from openasce.extension.debias.common.utils import DNNLayer
from openasce.extension.debias_model import CausalDebiasModel


class IPSDebiasModel(CausalDebiasModel):
    """Inverse Propensity Score Model of the causal debias.

    Model: IPS (Inverse Propensity Score)

    Paper: Estimating causal effects from large data sets using propensity scores[J].

    Author: Rubin, Donald B.
    """

    def __init__(
        self,
        hidden_units: Dict,
        min_propensity: float = 0.01,
        alpha: float = 0.1,
        lr: float = 0.1,
        name: str = "ips_debias",
    ) -> None:
        """Initialize.

        Args:
            hidden_units (dict): list of positive integer, the layer number and units in each layer.
            min_propensity (float): The minimum propensity at which to clip propensity estimates to avoid dividing by zero.
            alpha (float): hyperparameters of propensity loss
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.min_propensity = min_propensity
        self.alpha = alpha
        self.name = name
        self.base_embed_layer = DNNLayer(
            hidden_units=hidden_units["base"],
            activation=tf.nn.leaky_relu,
            apply_final_act=True,
            name="base_embed_layer",
        )

        self.treatment_embed_layer = DNNLayer(
            hidden_units=hidden_units["treatment"],
            activation=tf.nn.leaky_relu,
            apply_final_act=True,
            name="treatment_embed_layer",
        )

        self.outcome_pred_layer = DNNLayer(
            hidden_units=hidden_units["outcome"],
            activation=tf.nn.leaky_relu,
            apply_final_act=False,
            name="outcome_pred_layer",
        )

        self.propensity_pred_layer = DNNLayer(
            hidden_units=hidden_units["propensity"],
            activation=tf.nn.leaky_relu,
            apply_final_act=False,
            name="propensity_pred_layer",
        )
        self.optimizer = self.get_optimizer(lr=lr)

    @property
    def trainable_variables(self):
        variables = (
            self.base_embed_layer.trainable_variables
            + self.treatment_embed_layer.trainable_variables
            + self.outcome_pred_layer.trainable_variables
            + self.propensity_pred_layer.trainable_variables
        )
        return variables

    def forward(
        self, x: tf.Tensor, c: Dict[str, tf.Tensor], training: bool
    ) -> Dict[str, tf.Tensor]:
        base_emb = self.base_embed_layer(x)
        treatment = tf.reshape(c.get("treatment"), [-1, 1])
        t = tf.cast(treatment, dtype="float32")
        treatment_emb = self.treatment_embed_layer(t)
        outcome_emb = tf.concat([base_emb, treatment_emb], 1)
        outcome_logits = self.outcome_pred_layer(outcome_emb)
        propensity_logits = self.propensity_pred_layer(base_emb)
        predictions = {
            "outcome_logits": outcome_logits,
            "propensity_logits": propensity_logits,
        }
        return predictions

    def _call(
        self, *, x: tf.Tensor, y: tf.Tensor, c: Dict[str, tf.Tensor], training: bool
    ) -> Union[None, Dict[str, tf.Tensor]]:
        """
        Arguments:
            x: one batch of features
            y: one batch of labels, shape: [batch_size], outcome labels
            c: one batch for each concerned columns of the samples, here, {'treatment': Iterable[tf.Tensor]}
            training: True means training and False for predict
        Returns:
            None for training and Dict for predict
        """

        def grad(x, c, training, labels):
            with tf.GradientTape() as tape:
                predictions = self.forward(x, c, training)
                loss_value = self.loss(predictions, labels)
            return loss_value, tape.gradient(loss_value, self.trainable_variables)

        if training:
            y = tf.reshape(y, [-1, 1])
            t = tf.reshape(c.get("treatment"), [-1, 1])
            loss_value, grads = grad(x, c, training, [t, y])
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        else:
            predictions = self.forward(x, c, training)
            if y is not None:
                predictions["outcome"] = y
            return predictions

    def loss(self, predictions: Dict, labels: List[tf.Tensor]):
        outcome_logits, propensity_logits = (
            predictions["outcome_logits"],
            predictions["propensity_logits"],
        )
        treatment = tf.cast(labels[0], tf.int32)
        outcome_y = tf.cast(labels[1], tf.float32)
        treatment = tf.reshape(treatment, [-1])
        self.propensity_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=propensity_logits, labels=treatment
            )
        )
        user_propensity = tf.stop_gradient(tf.nn.softmax(propensity_logits, axis=1))
        t_onehot = tf.cast(tf.one_hot(treatment, user_propensity.shape[1]), tf.float32)
        propensity_score = tf.reduce_sum(tf.multiply(user_propensity, t_onehot), axis=1)
        propensity_score = tf.reshape(propensity_score, [-1, 1])

        # clip for inverse propensity score
        inverse_score = tf.minimum(1.0 / propensity_score, 1.0 / self.min_propensity)
        self.outcome_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=outcome_logits, labels=outcome_y
            )
            * inverse_score
        )
        loss = self.outcome_loss + self.alpha * self.propensity_loss
        return loss

    def get_optimizer(self, lr: float = 0.01):
        """Build the optimizer.

        Args:

        Returns:
            An optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        return optimizer
