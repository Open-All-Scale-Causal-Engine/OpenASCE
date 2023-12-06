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
from tensorflow.python.keras.layers import Dense

from openasce.extension.debias.common.utils import MultiTaskDNNModel
from openasce.extension.debias_model import CausalDebiasModel


class MACRDebiasModel(CausalDebiasModel):
    """Building a MACR model.

    Model: MACR (Model-Agnostic Counterfactual Reasoning).

    Paper: Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system.

    Link: https://arxiv.org/pdf/2010.15363.pdf.

    Author: Tianxin Wei, Fuli Feng, Jiawei Chen, Ziwei Wu, Jinfeng Yi and Xiangnan He.
    """

    def __init__(self, params: typing.Dict) -> None:
        super().__init__()
        # initialize params.
        self.hidden_units = params.get("hidden_units", [64, 16])
        self.task_hidden_units = params.get("task_hidden_units", [8])
        self.num_tasks = params.get("num_tasks", 2)
        self.act_fn = params.get("act_fn", "relu")
        self.l1_reg = params.get("l1_reg", 0.001)
        self.l2_reg = params.get("l2_reg", 0.001)
        self.dropout_rate = params.get("dropout_rate", 0)
        self.use_bn = params.get("use_bn", False)
        self.seed = params.get("seed", 1024)
        self.apply_final_act = params.get("apply_final_act", True)
        self.task_apply_final_act = params.get("task_apply_final_act", True)
        self.sigma = params.get("sigma", 0.001)
        self.alpha = params.get("alpha", 0.0001)
        self.beta = params.get("beta", 0.0001)
        self.lr = params.get("lr", 0.0001)

        # define model.
        self.model_user = MultiTaskDNNModel(
            hidden_units=self.hidden_units,
            task_hidden_units=self.task_hidden_units,
            num_tasks=self.num_tasks,
            act_fn=self.act_fn,
            l1_reg=self.l1_reg,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=self.apply_final_act,
            task_apply_final_act=self.task_apply_final_act,
            seed=self.seed,
            name="multi_dnn_model_user",
        )
        self.model_item = MultiTaskDNNModel(
            hidden_units=self.hidden_units,
            task_hidden_units=self.task_hidden_units,
            num_tasks=self.num_tasks,
            act_fn=self.act_fn,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=self.apply_final_act,
            task_apply_final_act=self.task_apply_final_act,
            seed=self.seed,
            name="multi_dnn_model_item",
        )
        self.optimizer = self.get_optimizer()

    @property
    def trainable_variables(self):
        variables = (
            self.model_user.trainable_variables + self.model_item.trainable_variables
        )
        return variables

    def forward(
        self, x: tf.Tensor, c: typing.Dict[str, tf.Tensor], training: bool
    ) -> typing.Dict[str, tf.Tensor]:
        user_feature, item_feature = c.get("user"), c.get("item")
        output_user = self.model_user(user_feature, training=training)
        output_item = self.model_item(item_feature, training=training)

        logits_user = Dense(1, activation=None, use_bias=True)(output_user[0])
        logits_item = Dense(1, activation=None, use_bias=True)(output_item[0])
        logits_hx = tf.reduce_sum(
            output_user[1] * output_item[1], axis=-1, keepdims=True
        )
        logit_out = (
            (logits_hx - self.sigma)
            * tf.nn.sigmoid(logits_user)
            * tf.nn.sigmoid(logits_item)
        )
        predictions = {
            "logit_out": logit_out,
            "logits_hx": logits_hx,
            "logits_user": logits_user,
            "logits_item": logits_item,
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
            c: the original input dict, here, {'user': np.ndarray, 'item': np.ndarray}.
                user: user feature.
                item: item feature.
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
            # train procedure.
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
        logits_hx = predictions["logits_hx"]
        logits_user = predictions["logits_user"]
        logits_item = predictions["logits_item"]

        loss_hx = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits_hx)
        loss_user = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits_user)
        loss_item = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits_item)
        # alpha，beta: hyper-parameters.
        alpha = self.alpha
        beta = self.beta
        loss = loss_hx + alpha * loss_user + beta * loss_item
        return loss

    def get_optimizer(self):
        """Build the optimizer.

        Args:

        Returns:
            An optimizer.
        """
        # lr: hyper-parameters.
        lr = self.lr
        return tf.keras.optimizers.Adam(lr=lr)
