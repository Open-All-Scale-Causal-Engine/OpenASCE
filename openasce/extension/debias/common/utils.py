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

from typing import List, Optional

import tensorflow as tf
from tensorflow.python.framework import tensor_shape

from openasce.utils.logger import logger


class DNNLayer(tf.keras.layers.Layer):
    """Building a MLP/DNN Layer.

    Layer: dense dnn layer

    inputs:
        2d tensor (batch_size, input_dim)
    outputs:
        2d tensor (batch_size, output_dim)
    """

    def __init__(
        self,
        hidden_units: Optional[List] = None,
        activation="relu",
        l1_reg=1e-4,
        l2_reg=1e-4,
        dropout_rate=0,
        use_bn=False,
        apply_final_act=True,
        seed=1024,
        **kwargs
    ):
        """Initialize DNNLayer.

        Args:
            hidden_units: list of positive integer, the layer number and units in each layer.
            activation: Activation function to use.
            l1_reg: float between 0 and 1. L1 regularizer strength applied to the kernel weights matrix.
            l2_reg: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
            dropout_rate: float in [0,1). Fraction of the units to dropout.
            use_bn: bool. Whether use BatchNormalization before activation or not.
            apply_final_act: whether to apply act in final layer
            seed: A Python integer to use as random seed.
        """
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.apply_final_act = apply_final_act
        self.hidden_outputs = []
        super(DNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)

        self.kernels = [
            self.add_weight(
                name="kernel" + str(i),
                shape=[hidden_units[i], hidden_units[i + 1]],
                initializer=tf.keras.initializers.he_normal(seed=self.seed),
                regularizer=tf.keras.regularizers.l1_l2(self.l1_reg, self.l2_reg),
                trainable=True,
            )
            for i in range(len(self.hidden_units))
        ]
        self.bias = [
            self.add_weight(
                name="bias" + str(i),
                shape=[
                    self.hidden_units[i],
                ],
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
            )
            for i in range(len(self.hidden_units))
        ]

        if self.use_bn:
            self.bn_layers = [
                tf.keras.layers.BatchNormalization(name="bn_layer_{}".format(i))
                for i in range(len(self.hidden_units))
            ]

        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.dropout_layers = [
                tf.keras.layers.Dropout(
                    self.dropout_rate,
                    seed=self.seed + i,
                    name="dropout_layer_{}".format(i),
                )
                for i in range(len(self.hidden_units))
            ]

        self.activation_layers = [
            tf.keras.layers.Activation(self.activation, name="act_layer_{}".format(i))
            for i in range(len(self.hidden_units))
        ]

        super(DNNLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        deep_output = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(
                tf.tensordot(deep_output, self.kernels[i], axes=(-1, 0)), self.bias[i]
            )
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            if i < len(self.hidden_units) - 1 or self.apply_final_act:
                fc = self.activation_layers[i](fc)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                fc = self.dropout_layers[i](fc, training=training)
            deep_output = fc
            self.hidden_outputs.append(fc)

        return deep_output

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, but saw: %s"
                % input_shape
            )
        elif len(self.hidden_units) > 0:
            return input_shape[:-1].concatenate(self.hidden_units[-1])
        else:
            return input_shape

    def get_config(self):
        config = {
            "activation": self.activation,
            "hidden_units": self.hidden_units,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "use_bn": self.use_bn,
            "dropout_rate": self.dropout_rate,
            "apply_final_act": self.apply_final_act,
            "seed": self.seed,
        }
        base_config = super(DNNLayer, self).get_config()
        config.update(base_config)
        return config


class DNNModel(tf.keras.Model):
    """Building a DNN model.

    Model: DNN or MLP.

    inputs:
        2d tensor (batch_size, input_dim).
    outputs:
        2d tensor (batch_size, output_dim).
    """

    def __init__(
        self,
        hidden_units,
        act_fn="relu",
        l1_reg=1e-4,
        l2_reg=1e-4,
        dropout_rate=0,
        use_bn=False,
        seed=1024,
        apply_final_act=False,
        name="DNNModel",
    ):
        """Initialize DNNModel.

        Args:
            hidden_units: list, unit in each hidden layer.
            act_fn: string, activation function.
            l2_reg: float, regularization value.
            dropout_rate: float, fraction of the units to dropout.
            use_bn: boolean, if True, apply BatchNormalization in each hidden layer.
            seed: int, random value for initialization.

        """
        super(DNNModel, self).__init__(name="DNNModel")
        self.dnn_layer = DNNLayer(
            hidden_units=hidden_units,
            activation=act_fn,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            apply_final_act=apply_final_act,
            seed=seed,
            name="{}_dnn_layer".format(name),
        )

    def call(self, inputs, training=None):
        """Calls the model on new inputs.

        Args:
            inputs: 2d tensor (batch_size, dim_1), deep features.
            wide_input: 2d tensor (batch_size, dim_2), wide features.

        Returns:
            2d tensor (batch_size, out_dim).

        """
        dnn_output = self.dnn_layer(inputs, training=training)
        return dnn_output


class MultiTaskDNNModel(tf.keras.Model):
    """Building a multi task dnn model.

    Model: MultiTaskDNN.

    inputs:
        2d tensor (batch_size, input_dim).
    outputs:
         list odf 2d tensor [(batch_size, output_dim),..].
    """

    def __init__(
        self,
        hidden_units,
        num_tasks,
        task_hidden_units,
        act_fn="relu",
        l1_reg=1e-4,
        l2_reg=1e-4,
        dropout_rate=0,
        use_bn=False,
        seed=1024,
        apply_final_act=False,
        task_apply_final_act=False,
        name="MultiTaskDNNModel",
    ):
        """Initialize MultiTaskDNNModel.

        Args:
            hidden_units: list, unit in each hidden layer.
            act_fn: string, activation function.
            num_tasks: int, number of the task.
            l2_reg: float, regularization value.
            dropout_rate: float, fraction of the units to dropout.
            use_bn: boolean, if True, apply BatchNormalization in each hidden layer.
            seed: int, random value for initialization.
        """
        super(MultiTaskDNNModel, self).__init__(name="MultiTaskDNNModel")
        self.num_tasks = num_tasks
        self.dnn_layer = DNNLayer(
            hidden_units=hidden_units,
            activation=act_fn,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            apply_final_act=apply_final_act,
            seed=seed,
            name="{}_dnn_layer".format(name),
        )
        self.task_layers = []
        for i in range(self.num_tasks):
            self.task_layers.append(
                DNNLayer(
                    hidden_units=task_hidden_units,
                    activation=act_fn,
                    l1_reg=l1_reg,
                    l2_reg=l2_reg,
                    dropout_rate=dropout_rate,
                    use_bn=use_bn,
                    apply_final_act=task_apply_final_act,
                    seed=seed,
                )
            )

    def call(self, inputs, training=None):
        """Calls the model on new inputs.

        Args:
            inputs: 2d tensor (batch_size, dim_1), deep features.
            wide_input: 2d tensor (batch_size, dim_2), wide features.

        Returns:
            list of 2d tensor [(batch_size, output_dim),..].

        """
        task_outputs = []
        task_inputs = self.dnn_layer(inputs, training=training)

        for i in range(self.num_tasks):
            task_outputs.append(self.task_layers[i](task_inputs, training=training))
        return task_outputs


class FMLayer(tf.keras.layers.Layer):
    """Building a FM Layer.
    Model: Factorization Machine.

    Paper: Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.

    Link: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf.

    Author: Steffen Rendle

    Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
    Output shape
        - 2D tensor with shape: ``(batch_size, dim)``.

    """

    def __init__(self, **kwargs):
        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions % d, expect to be 3 dimensions"
                % (len(input_shape))
            )

        # Be sure to call this somewhere!
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Calls the layer on new inputs.
        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples (one per output tensor of the layer).

        """
        if tf.keras.backend.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (tf.keras.backend.ndim(inputs))
            )

        fm_inputs = inputs
        squared_sum = tf.keras.backend.square(
            tf.keras.backend.sum(fm_inputs, axis=1, keepdims=True)
        )
        # squared_sum (batch, dim)
        logger.debug("squared_sum shape {}".format(squared_sum.shape))
        sum_squared = tf.keras.backend.sum(fm_inputs * fm_inputs, axis=1, keepdims=True)
        logger.debug("sum_squared shape {}".format(sum_squared.shape))
        output = tf.keras.layers.Lambda(lambda x: 0.5 * tf.subtract(x[0], x[1]))(
            [squared_sum, sum_squared]
        )
        output = tf.keras.backend.squeeze(output, axis=1)
        logger.debug("output {}".format(output))
        return output

    def compute_output_shape(self, input_shape):
        # return (input_shape[0], input_shape[-1])
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, but saw: %s"
                % input_shape
            )
        return input_shape[:1].concatenate(input_shape[-1])


SQRT_CONST = 1e-3


def mmd_rbf(X, t, p, sig):
    """Computes the l2-RBF MMD for X given t

    Paper: Estimating individual treatment effect: generalization bounds and algorithms.
    Gaussian kernel(RBF): https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    Args:
        X: embeddings.
        t: 1 or 0. Distinguish the treatment group and the control group.
        p: the proportion of the number of treatment instances / the number of all instances.
        sig: kernel width.

    """

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]

    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)

    Kcc = tf.exp(-pdist2sq(Xc, Xc) / tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc, Xt) / tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt, Xt) / tf.square(sig))

    m = tf.compat.v1.to_float(tf.shape(Xc)[0])
    n = tf.compat.v1.to_float(tf.shape(Xt)[0])

    mmd = tf.square(1.0 - p) / (m * (m - 1.0)) * (tf.reduce_sum(Kcc) - m)
    mmd = mmd + tf.square(p) / (n * (n - 1.0)) * (tf.reduce_sum(Ktt) - n)
    mmd = mmd - 2.0 * p * (1.0 - p) / (m * n) * tf.reduce_sum(Kct)
    mmd = 4.0 * mmd

    return mmd


def pdist2sq(X, Y):
    """Computes the squared Euclidean distance between all pairs x in X, y in Y"""
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.compat.v1.reduce_sum(tf.square(X), 1, keep_dims=True)
    ny = tf.compat.v1.reduce_sum(tf.square(Y), 1, keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D
