import os

import numpy as np
import tensorflow as tf
from sklearn import model_selection

from openasce.extension.debias import IPSDebiasModel
from openasce.utils.logger import logger


def get_rossi_dataset():
    train_dataset_url = "https://github.com/CamDavidsonPilon/lifelines/raw/master/lifelines/datasets/rossi.csv"
    # column order in CSV file
    column_names = [
        "week",
        "arrest",
        "fin",
        "age",
        "race",
        "wexp",
        "mar",
        "paro",
        "prio",
    ]
    feature_names = [column_names[0]] + column_names[2:]
    label_name = column_names[1]
    logger.info("Features: {}".format(feature_names))
    logger.info("Label: {}".format(label_name))
    train_dataset_path = tf.keras.utils.get_file(
        fname=os.path.basename(train_dataset_url), origin=train_dataset_url
    )
    logger.info(f"Local copy of the dataset file: {train_dataset_path}")
    batch_size = 100
    dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_path,
        batch_size=batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1,
    )

    def pack_features_vector(features, labels):
        treatmeat = tf.cast(features["week"] >= 30, tf.int32)
        features = tf.stack(list([features[i] for i in feature_names[1:]]), axis=1)
        return tf.cast(features, tf.float32), labels, {"treatment": treatmeat}

    dataset = dataset.shuffle(buffer_size=10 * batch_size)
    dataset = dataset.map(pack_features_vector)
    test_size = 1
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size).shuffle(buffer_size=200, seed=1)

    return train_dataset, test_dataset


def test_model():
    tf.compat.v1.enable_eager_execution()
    train_dataset, test_dataset = get_rossi_dataset()
    model = IPSDebiasModel(
        hidden_units={
            "base": [16],
            "treatment": [8],
            "outcome": [16, 1],
            "propensity": [8, 2],
        },
        min_propensity=0.01,
        lr=0.1,
    )
    model.fit(Z=train_dataset, num_epochs=200)
    model.predict(Z=test_dataset)
    result = model.get_result()
    outcome_scores = tf.sigmoid(result["outcome_logits"])
    from sklearn.metrics import roc_auc_score

    cvr_auc = roc_auc_score(result["outcome"], outcome_scores)
    logger.info("auc: {:.4f}".format(cvr_auc))


if __name__ == "__main__":
    test_model()
