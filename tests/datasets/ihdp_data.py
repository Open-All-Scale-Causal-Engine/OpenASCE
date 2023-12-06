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

import inspect
import os

import pandas as pd

from openasce.utils.logger import logger


def get_ihdp_data():
    """
    Loads the IHDP dataset, refer to https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv
    """
    col_names = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [
        "x{}".format(i + 1) for i in range(25)
    ]
    csv_path = os.path.join(os.path.dirname(inspect.getfile(get_ihdp_data)), "ihdp.csv")
    df = pd.read_csv(csv_path, names=col_names)
    logger.info("IHDP dataset loaded.")
    return df.iloc[:523], df.iloc[523:]
