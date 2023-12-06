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
from openasce.utils.logger import logger


class DRDebiasModel(CausalDebiasModel):
    """Doubly Robust Model for Debiasing Exposure/Post-click.

    Model: DR (Doubly robust)

    Paper: Doubly robust joint learning for recommendation on data missing not at random[C].

    Link: https://proceedings.mlr.press/v97/wang19n.html.

    Author: Wang X, Zhang R, Sun Y, et al.
    """

    def __init__(
        self,
        hidden_units: Dict,
        min_propensity: float = 0.01,
        t_is_multi_class: bool = False,
        t_as_feature: bool = False,
        lr: float = 0.1,
        name: str = "dr_debias",
    ) -> None:
        """Initialize.

        Args:
            hidden_units (dict): list of positive integer, the layer number and units in each layer.
            min_propensity (float): The minimum propensity at which to clip propensity estimates to avoid dividing by zero.
            t_is_multi_class (bool): wether the treatment label is multi-class.
            t_as_feature (bool): wether the treatment is observed feature.
            lr (float): learning rate
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.min_propensity = min_propensity
        self.t_is_multi_class = t_is_multi_class
        self.t_as_feature = t_as_feature
        self.name = name
        self.base_embed_layer = DNNLayer(
            hidden_units=hidden_units["base"],
            activation=tf.nn.leaky_relu,
            apply_final_act=True,
            name="base_pred_layer",
        )

        self.ctr_pred_layer = DNNLayer(
            hidden_units=hidden_units["ctr"],
            activation=tf.nn.leaky_relu,
            apply_final_act=False,
            name="ctr_pred_layer",
        )

        self.cvr_pred_layer = DNNLayer(
            hidden_units=hidden_units["cvr"],
            activation=tf.nn.leaky_relu,
            apply_final_act=False,
            name="cvr_pred_layer",
        )

        self.dr_pred_layer = DNNLayer(
            hidden_units=hidden_units["dr"],
            activation=tf.nn.leaky_relu,
            apply_final_act=False,
            name="dr_pred_layer",
        )

        if self.t_as_feature:
            self.t_encode_layer = DNNLayer(
                hidden_units=hidden_units["t"],
                activation=tf.nn.leaky_relu,
                apply_final_act=True,
                name="t_encode_layer",
            )
        self.optimizer = self.get_optimizer(lr=lr)

    @property
    def trainable_variables(self):
        variables = (
            self.base_embed_layer.trainable_variables
            + self.ctr_pred_layer.trainable_variables
            + self.cvr_pred_layer.trainable_variables
            + self.dr_pred_layer.trainable_variables
        )
        if self.t_as_feature:
            variables += self.t_encode_layer.trainable_variables
        return variables

    def forward(
        self, x: tf.Tensor, c: Dict[str, tf.Tensor], training: bool
    ) -> Dict[str, tf.Tensor]:
        inputs = self.base_embed_layer(x)
        t = tf.reshape(c.get("treatment"), [-1, 1])
        ctr_logits = self.ctr_pred_layer(inputs)
        if self.t_as_feature:
            t = tf.cast(t, tf.float32)
            t_features = self.t_encode_layer(t)
            inputs = tf.concat([inputs, t_features], -1)
        cvr_logits = self.cvr_pred_layer(inputs)
        dr_logits = self.dr_pred_layer(inputs)
        predictions = {
            "ctr_logits": ctr_logits,
            "cvr_logits": cvr_logits,
            "dr_logits": dr_logits,
        }
        return predictions

    def _call(
        self, *, x: tf.Tensor, y: tf.Tensor, c: Dict[str, tf.Tensor], training: bool
    ) -> Union[None, Dict[str, tf.Tensor]]:
        """
        Arguments:
            x: one batch of features
            y: one batch of labels, shape: [batch_size, 2], including ctr and cvr labels
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
                predictions["cvr_labels"] = y
            return predictions

    def loss(self, predictions: Dict, labels: List[tf.Tensor]):
        ctr_logits, cvr_logits, dr_logits = (
            predictions["ctr_logits"],
            predictions["cvr_logits"],
            predictions["dr_logits"],
        )
        ctr_y = tf.cast(labels[0], tf.float32)
        cvr_y = tf.cast(labels[1], tf.float32)
        cvr_label = tf.stop_gradient(tf.sigmoid(cvr_logits))
        if not self.t_is_multi_class:
            self.propensity_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=ctr_logits, labels=ctr_y)
            )
            propensity_score = tf.maximum(
                tf.stop_gradient(tf.sigmoid(ctr_logits)), self.min_propensity
            )
        else:
            ctr_label = tf.cast(tf.reshape(ctr_y, [-1]), tf.int64)
            self.propensity_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=ctr_logits, labels=ctr_label
                )
            )
            propensity_score = tf.gather(
                tf.maximum(
                    tf.stop_gradient(tf.nn.softmax(ctr_logits, -1)),
                    tf.convert_to_tensor(self.min_propensity, dtype="float32"),
                ),
                ctr_label,
                axis=1,
            )

        self.impute_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=cvr_logits, labels=cvr_y)
            / propensity_score
        )
        hat_e_ui = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=dr_logits, labels=cvr_label
        )
        e_ui = tf.nn.sigmoid_cross_entropy_with_logits(logits=dr_logits, labels=cvr_y)
        if self.t_is_multi_class:
            ctr_y = tf.cast(ctr_y > 0, tf.float32)
        self.dr_loss = tf.reduce_mean(
            tf.multiply(e_ui - hat_e_ui, tf.compat.v1.div(ctr_y, propensity_score))
            + hat_e_ui
        )
        loss = self.propensity_loss + self.impute_loss + self.dr_loss
        logger.info(
            f"loss_value: {loss}=[{self.propensity_loss}+{self.impute_loss}+{self.dr_loss}]"
        )
        return loss

    def get_optimizer(self, lr: float = 0.01):
        """Build the optimizer.

        Args:

        Returns:
            An optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        return optimizer
