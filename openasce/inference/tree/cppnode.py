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

from typing import List

from .gbct_utils import common


def create_didnode_from_dict(info):
    """
    Create a CppDebiasNode from a dictionary.

    Arguments:
        info (Dict): The node information.

    Returns:
        CppDebiasNode: The CppDebiasNode instance.
    """
    # basic information for tree node
    assert len(info['children']) == 2, f'children should be 2!'
    node = common.CppDebiasNode()
    basic_keys = node.get_property_keys()
    for k in info.keys():
        if k in basic_keys:
            setattr(node, k, info[k])
        else:
            node.set_info(k, info[k])
    return node


def predict(nodes: List[common.CppDiDNode], x, out, key, threads=20):
    """
    Predict using the tree nodes.

    Arguments:
        nodes (List): The list of tree nodes.
        x (ndarray): The input data.
        out (ndarray): The output array.
        key (ndarray): The prediction key.
        threads (int): The number of threads.

    Returns:
        ndarray: The predicted values.

    Raises:
        RuntimeError: If the number of nodes is less than or equal to 0.
        ValueError: If the node type is not supported.
    """
    if len(nodes) <= 0:
        raise RuntimeError(f'The number of nodes must be greater than 0!')
    elif isinstance(nodes[0], list) is False:
        nodes = [nodes]
    if isinstance(nodes[0][0], common.CppDiDNode):
        return common.predict_did(nodes, out, x, key, threads)
    elif isinstance(nodes[0][0], common.CppDebiasNode):
        return common.predict_debias(nodes, out, x, key, threads)
    else:
        raise ValueError(f'{type(nodes[0][0])} is not supported!')
