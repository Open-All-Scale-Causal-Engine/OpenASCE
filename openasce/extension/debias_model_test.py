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

import os
from typing import Dict
from unittest import TestCase

import numpy as np
import tensorflow as tf

from openasce.extension.debias_model import CausalDebiasModel
from openasce.utils.logger import logger


def get_iris_dataset():
    train_dataset_url = (
        "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    )

    train_dataset_path = tf.keras.utils.get_file(
        fname=os.path.basename(train_dataset_url), origin=train_dataset_url
    )
    logger.info(f"Local copy of the train dataset file: {train_dataset_path}")
    test_url = (
        "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
    )
    test_dataset_path = tf.keras.utils.get_file(
        fname=os.path.basename(test_url), origin=test_url
    )
    logger.info(f"Local copy of the test dataset file: {test_dataset_path}")
    # column order in CSV file
    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    feature_names = column_names[:-1]
    label_name = column_names[-1]
    logger.info("Features: {}".format(feature_names))
    logger.info("Label: {}".format(label_name))

    batch_size = 32
    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_path,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1,
    )
    test_dataset = tf.data.experimental.make_csv_dataset(
        test_dataset_path,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1,
    )

    def pack_features_vector(features, labels):
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    train_dataset = train_dataset.map(pack_features_vector)
    test_dataset = test_dataset.map(pack_features_vector)
    return (train_dataset, test_dataset)


class MockDebiasModel(CausalDebiasModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
                tf.keras.layers.Dense(10, activation=tf.nn.relu),
                tf.keras.layers.Dense(3),
            ]
        )
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    def _call(
        self, *, x: np.ndarray, y: np.ndarray, c: Dict[str, np.ndarray], training: bool
    ) -> None:
        def loss(x, y):
            y_ = self.model(x, training=True)
            return self.loss_object(y_true=y, y_pred=y_)

        def grad(inputs, targets):
            with tf.GradientTape() as tape:
                loss_value = loss(inputs, targets)
            return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

        x = x if x is not None else c.get("X")  # For testcase not having X or Y
        if training:
            y = y if y is not None else c.get("Y")  # For testcase not having X or Y
            loss_value, grads = grad(x, y)
            logger.info(f"loss_value={loss_value}")
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        else:
            r = self.model(x, training=False).numpy()
            logger.info(f"r.shape={r.shape}")
            return {"result": r[:, 0], "other_result": r[:, 0]}


class TestRuntime(TestCase):
    def setUp(self) -> None:
        pass

    def test_debias_model_execution(self) -> None:
        r = MockDebiasModel()
        (train_dataset, test_dataset) = get_iris_dataset()
        features, labels = next(iter(train_dataset))
        train_data = features.numpy()
        predict_data = labels.numpy()
        mock_c = {"X": [train_data, train_data], "Y": [predict_data, predict_data]}
        r.fit(X=mock_c.get("X"), Y=mock_c.get("Y"), C=mock_c, num_epochs=5)
        r.predict(X=mock_c.get("X"), C=mock_c)
        result = r.get_result()

    def test_non_x_y_model_execution(self) -> None:
        r = MockDebiasModel()
        (train_dataset, test_dataset) = get_iris_dataset()
        features, labels = next(iter(train_dataset))
        train_data = features.numpy()
        predict_data = labels.numpy()
        mock_c = {"X": [train_data, train_data], "Y": [predict_data, predict_data]}
        r.fit(C=mock_c, num_epochs=3)
        r.predict(X=mock_c.get("X"), C=mock_c)
        result = r.get_result()

    def test_z_model_execution(self) -> None:
        r = MockDebiasModel()
        (train_dataset, test_dataset) = get_iris_dataset()
        r.fit(Z=train_dataset, num_epochs=5)
        r.predict(Z=test_dataset)
        result = r.get_result()
        logger.info(f"result={result}")
