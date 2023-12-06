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

from openasce.extension.debias.common.utils import DNNModel
from openasce.extension.debias_model import CausalDebiasModel


class DICEDebiasModel(CausalDebiasModel):
    """Building a DICE model.

    Model: DICE (Disentangling Interest and Conformity with Causal Embedding).

    Link: https://arxiv.org/pdf/2006.11011.pdf.

    Author: Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Depeng Jin, Yong Li.

    """

    def __init__(self, params: typing.Dict) -> None:
        """Initialize.

        Args:
            params: parameter dict.
        """
        super().__init__()
        # initialize params.
        self.hidden_units = params.get("hidden_units", [64, 16])
        self.act_fn = params.get("act_fn", "relu")
        self.l2_reg = params.get("l2_reg", 0.001)
        self.dropout_rate = params.get("dropout_rate", 0)
        self.use_bn = params.get("use_bn", False)
        self.apply_final_act = params.get("apply_final_act", True)
        self.lr = params.get("lr", 0.0001)
        self.alpha = params.get("alpha", 0.1)
        self.beta = params.get("beta", 0.01)

        # define model.
        self.model_interest = DNNModel(
            hidden_units=self.hidden_units,
            act_fn=self.act_fn,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=self.apply_final_act,
            name="dnn_model_interest",
        )

        self.model_conformity = DNNModel(
            hidden_units=self.hidden_units,
            act_fn=self.act_fn,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=self.apply_final_act,
            name="dnn_model_conformity",
        )

        self.model_cilck = DNNModel(
            hidden_units=[1024, 128, 1],
            act_fn=self.act_fn,
            l2_reg=self.l2_reg,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            apply_final_act=False,
            name="dnn_model_click",
        )

        self.optimizer = self.get_optimizer()

    @property
    def trainable_variables(self):
        variables = (
            self.model_interest.trainable_variables
            + self.model_conformity.trainable_variables
            + self.model_cilck.trainable_variables
        )
        return variables

    def forward(
        self, x: tf.Tensor, c: typing.Dict[str, tf.Tensor], training: bool
    ) -> typing.Dict[str, tf.Tensor]:
        feature, mask = c.get("feature"), c.get("mask")
        interest_emb = self.model_interest(feature, training=training)
        conformity_emb = self.model_conformity(feature, training=training)
        concat_emb = tf.concat([interest_emb, conformity_emb], axis=1)

        logit_interest = Dense(1, activation=None, use_bias=True)(interest_emb)
        logit_conformity = Dense(1, activation=None, use_bias=True)(conformity_emb)
        logit_click = self.model_cilck(concat_emb, training=training)

        predictions = {
            "logit_interest": logit_interest,
            "logit_conformity": logit_conformity,
            "logit_click": logit_click,
            "conformity_emb": conformity_emb,
            "interest_emb": interest_emb,
            "mask": mask,
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
            c: the original input dict, here, {'feature': np.ndarray, 'mask': np.ndarray}.
                feature: train feature.
                mask: indicate the instance belong to interest or conformity, 1 means interest, 0 means conformity.
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
            predictions:  a dictionary holding prediction tensors, where
                logits_interest: model output logits tensor.
                logits_conformity: model output logits tensor.
                logits_click: model output logits tensor.
                conformity_emb: model output conformity_emb tensor.
                interest_emb: model output interest_emb tensor.
                mask: indicate the instance belong to interest or conformity, 1 means interest, 0 means conformity.
            labels: label tensor dict.

        Returns:
            A scalar loss or A dictionary mapping strings (loss names) to scalar loss.
        """
        logit_click = predictions["logit_click"]
        logit_interest = predictions["logit_interest"]
        logit_conformity = predictions["logit_conformity"]
        conformity_emb = predictions["conformity_emb"]
        interest_emb = predictions["interest_emb"]
        mask = predictions["mask"]

        loss_click = tf.compat.v1.losses.sigmoid_cross_entropy(labels, logit_click)
        loss_interest = tf.compat.v1.losses.sigmoid_cross_entropy(
            labels, logit_interest, weights=mask
        )
        loss_conformity = tf.compat.v1.losses.sigmoid_cross_entropy(
            labels, logit_conformity, weights=1 - mask
        )
        loss_discrepancy = tf.compat.v1.losses.mean_squared_error(
            conformity_emb, interest_emb
        )

        # alpha，beta: hyper-parameters.
        alpha = self.alpha
        beta = self.beta
        loss = (
            loss_click
            + alpha * (loss_interest + loss_conformity)
            - beta * loss_discrepancy
        )
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
