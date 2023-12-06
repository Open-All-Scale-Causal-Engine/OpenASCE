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

import tensorflow as tf
from sklearn.metrics import roc_auc_score

from openasce.extension.debias import MACRDebiasModel
from openasce.utils.logger import logger


class ParamFactory(object):
    def __init__(self):
        super(ParamFactory, self).__init__()
        self.params = {
            "model_name": "dnn_base",
            # default
            "hidden_units": [1024],
            "task_hidden_units": [128, 16],
            "num_tasks": 2,
            "apply_final_act": True,
            "task_apply_final_act": True,
            "act_fn": "relu",
            "l2_reg": 0.001,
            "dropout_rate": 0,
            "use_bn": False,
            "lr": 0.0001,
            "sigma": 0.001,
            "alpha": 0.01,
            "beta": 0.01,
            "batch_size": 1000,
            "epoch": 1,
            "seeds": 1024,
            # must be defined by user
            "column_names": [
                "f1",
                "f2",
                "f3",
                "f4",
                "f5",
                "f6",
                "f7",
                "f8",
                "f9",
                "f10",
                "f11",
                "f12",
                "f13",
                "f14",
                "f15",
                "y",
                "T",
                "W",
                "G",
            ],
            "train_dataset_path": "../data/data.csv",
        }

    def make_params(self, model_tag=None):
        return self.params


params = ParamFactory().make_params()


def get_dataset():
    train_dataset_path = params.get("train_dataset_path", None)
    column_names = params.get("column_names", None)
    label_name = column_names[15]
    user_feature_names = column_names[0:10]
    item_feature_names = column_names[10:15]

    logger.info("User Features: {}".format(user_feature_names))
    logger.info("Item Features: {}".format(item_feature_names))
    logger.info("Label: {}".format(label_name))
    logger.info(f"Local copy of the dataset file: {train_dataset_path}")

    batch_size = params.get("batch_size", 50)
    num_epochs = params.get("num_epochs", 1)
    seeds = params.get("seeds", 1024)
    dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_path,
        batch_size=batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=num_epochs,
    )

    def pack_features_vector(features, labels):
        user = tf.stack(list([features[i] for i in user_feature_names[:-1]]), axis=1)
        item = tf.stack(list([features[i] for i in item_feature_names[:-1]]), axis=1)
        labels = tf.reshape(labels, [batch_size, 1])
        return tf.cast(user, tf.float32), labels, {"user": user, "item": item}

    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.map(pack_features_vector)
    test_size = 1
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size).shuffle(
        buffer_size=10 * batch_size, seed=seeds
    )

    return train_dataset, test_dataset


def test_model():
    tf.compat.v1.enable_eager_execution()
    train_dataset, test_dataset = get_dataset()
    model = MACRDebiasModel(params)
    model.fit(Z=train_dataset, num_epochs=10)
    model.predict(Z=test_dataset)
    result = model.get_result()
    scores = tf.sigmoid(result["logits_hx"])
    out_auc = roc_auc_score(result["labels"], scores)
    logger.info("auc: {:.4f}".format(out_auc))
    assert out_auc > 0.5


if __name__ == "__main__":
    test_model()
