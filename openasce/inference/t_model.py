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

from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import tensorflow as tf

from openasce.extension.debias.common.utils import DNNLayer
from openasce.inference.inference_model import InferenceModel
from openasce.utils.logger import logger


class TModel(InferenceModel):
    """
    T_model based on NN
    """

    def __init__(
        self, hidden_units: Dict, lr: float = 0.1, name: str = "t_model"
    ) -> None:
        """Initialize.

        Args:
            hidden_units (dict): list of positive integer, the layer number and units in each layer.
            lr (float): learning rate
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.name = name
        self.base_embed_layer = DNNLayer(
            hidden_units=hidden_units["base"],
            activation=tf.nn.leaky_relu,
            apply_final_act=True,
            name="base_embed_layer",
        )

        self.test_embed_layer = DNNLayer(
            hidden_units=hidden_units["test"],
            activation=tf.nn.leaky_relu,
            apply_final_act=True,
            name="test_embed_layer",
        )

        self.control_embed_layer = DNNLayer(
            hidden_units=hidden_units["control"],
            activation=tf.nn.leaky_relu,
            apply_final_act=False,
            name="control_embed_layer",
        )

        self.optimizer = self.get_optimizer(lr=lr)

    def fit(
        self,
        X: Iterable[tf.Tensor] = None,
        Y: Iterable[tf.Tensor] = None,
        T: Iterable[tf.Tensor] = None,
        *,
        Z: Iterable[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = None,
        num_epochs: int = 1,
        **kwargs,
    ) -> None:
        """Feed the sample data and train the model on the samples.

        Arguments:
            X: Features of the samples.
            Y: Outcomes of the samples.
            T: Treatments of the samples.
            Z: The iterable object returning (a batch of X, a batch of Y, a batch of T)
            num_epochs: number of the train epoch
        Returns:
            None
        """
        self._X = X
        self._Y = Y
        self._T = T
        self._Z = Z
        self._train_loop(num_epochs=num_epochs, **kwargs)

    def estimate(
        self,
        X: Iterable[tf.Tensor] = None,
        T: Iterable[tf.Tensor] = None,
        *,
        Z: Iterable[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = None,
        **kwargs,
    ) -> None:
        """Feed the sample data
        Estimate the effect on the samples, and get_result method can be used to get the result of prediction
        Arguments:
            X: Features of the samples.
            T: Treatments of the samples.
            Z: The iterable object returning (a batch of X, a batch of Y, a batch of T)
        Returns:
            None
        """
        self._X = X
        self._T = T
        self._Z = Z
        self._estimate_result = self._predict_loop(**kwargs)

    def _predict_loop(self):
        """main loop for prediction"""
        f_result = {}
        for z in self._generator():
            result = self._call(
                x=z[0] if len(z) > 0 else None,
                y=z[1] if len(z) > 1 else None,
                t=z[2] if len(z) > 2 else None,
                training=False,
            )
            for k, v in result.items():
                f_result[k] = np.hstack([f_result[k], v]) if k in f_result else v
        return f_result

    def _train_loop(self, *, num_epochs, **kwargs):
        """main loop for train"""

        curr_epoch = 0
        while curr_epoch < num_epochs:
            for z in self._generator():
                self._call(
                    x=z[0] if len(z) > 0 else None,
                    y=z[1] if len(z) > 1 else None,
                    t=z[2] if len(z) > 2 else None,
                    training=True,
                )
            logger.info(f"Finish epoch {curr_epoch}.")
            curr_epoch += 1

    @property
    def trainable_variables(self):
        variables = (
            self.base_embed_layer.trainable_variables
            + self.test_embed_layer.trainable_variables
            + self.control_embed_layer.trainable_variables
        )
        return variables

    def forward(
        self, x: tf.Tensor, t: tf.Tensor, training: bool
    ) -> Dict[str, tf.Tensor]:
        base_emb = self.base_embed_layer(x)
        test_logits = self.test_embed_layer(base_emb)
        control_logits = self.control_embed_layer(base_emb)
        treatment = tf.cast(tf.reshape(t, [-1, 1]), dtype="float32")
        treatment = tf.concat([1 - treatment, treatment], axis=1)
        logits = tf.concat([control_logits, test_logits], axis=1)
        outcome_logits = tf.reshape(
            tf.reduce_sum(tf.multiply(logits, treatment), axis=1), [-1, 1]
        )
        predictions = {
            "test_logits": test_logits,
            "control_logits": control_logits,
            "outcome_logits": outcome_logits,
        }
        return predictions

    def _call(
        self, *, x: tf.Tensor, y: tf.Tensor, t: tf.Tensor, training: bool
    ) -> Union[None, Dict[str, tf.Tensor]]:
        """
        Arguments:
            x: one batch of features
            y: one batch of labels, shape: [batch_size], outcome labels
            t: one batch of treatments
            training: True means training and False for predict
        Returns:
            None for training and Dict for predict
        """

        def grad(x, t, training, labels):
            with tf.GradientTape() as tape:
                predictions = self.forward(x, t, training)
                loss_value = self.loss(predictions, labels)
            return loss_value, tape.gradient(loss_value, self.trainable_variables)

        if training:
            y = tf.reshape(y, [-1, 1])
            t = tf.reshape(t, [-1, 1])
            loss_value, grads = grad(x, t, training, [t, y])
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        else:
            predictions = self.forward(x, t, training)
            if y is not None:
                predictions["outcome"] = y
            return predictions

    def loss(self, predictions: Dict, labels: List[tf.Tensor]):
        control_logits, test_logits = (
            predictions["control_logits"],
            predictions["test_logits"],
        )
        treatment = tf.cast(labels[0], tf.float32)
        outcome_y = tf.cast(labels[1], tf.float32)
        treatment = tf.concat([1 - treatment, treatment], 1)
        logits = tf.concat([control_logits, test_logits], 1)
        outcome_logits = tf.reshape(
            tf.reduce_sum(tf.multiply(logits, treatment), axis=1), [-1, 1]
        )
        self.outcome_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=outcome_logits, labels=outcome_y
            )
        )
        return self.outcome_loss

    def get_optimizer(self, lr: float = 0.01):
        """Build the optimizer.

        Args:

        Returns:
            An optimizer.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        return optimizer

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
