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

import logging

GLOBAL_LOGGER_NAME = "openasce-log"
DEFAULT_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)
DEFAULT_FORMATTER = logging.Formatter(DEFAULT_FORMAT)


def init_custom_logger(name):
    logger = logging.getLogger(name)
    ch = logging.StreamHandler()
    ch.setFormatter(DEFAULT_FORMATTER)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class openasceLogger(object):
    log_instance = init_custom_logger(name=GLOBAL_LOGGER_NAME)


logger = openasceLogger.log_instance
