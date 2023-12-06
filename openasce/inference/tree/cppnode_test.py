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


import unittest as ut
from time import time

import numpy as np

from openasce.inference.tree.cppnode import create_didnode_from_dict, predict


class TestCPPDiDNode(ut.TestCase):
    def test_createCPPDiDNode(self):
        arr = np.arange(10).reshape([2, 5])
        info = {
            "children": [1, 2],
            "split_feature": 1,
            "split_thresh": 1,
            "leaf_id": 0,
            "level_id": 0,
            "is_leaf": True,
            "outcomes": arr,
            "bias": arr,
        }
        node = create_didnode_from_dict(info)
        self.assertEqual(node.split_feature, info["split_feature"])
        self.assertEqual(node.split_thresh, info["split_thresh"])
        self.assertEqual(node.is_leaf, info["is_leaf"])
        self.assertListEqual(node.children, info["children"])
        self.assertTrue(np.array_equal(node.outcomes, arr))
        self.assertTrue(np.array_equal(node.bias, arr))

    def test_predict(self):
        n, m, k = 100000, 10, 2
        arr = np.arange(2 * k).reshape([2, k])
        infos = [
            {
                "children": [1, 2],
                "split_feature": 1,
                "split_thresh": 0,
                "leaf_id": 0,
                "level_id": 0,
                "is_leaf": False,
                "outcomes": np.full_like(arr, 1),
                "bias": arr,
            },
            {
                "children": [-1, -1],
                "split_feature": 0,
                "split_thresh": 1,
                "leaf_id": 1,
                "level_id": 1,
                "is_leaf": True,
                "outcomes": np.full_like(arr, 1),
                "bias": arr,
            },
            {
                "children": [-1, -1],
                "split_feature": 2,
                "split_thresh": 1,
                "leaf_id": 2,
                "level_id": 1,
                "is_leaf": True,
                "outcomes": np.full_like(arr, 2),
                "bias": arr,
            },
        ]
        x = np.zeros([n, m])
        out = np.zeros([n, 1, 2, k])
        nodes = [create_didnode_from_dict(info) for info in infos]
        st = time()
        predict(nodes, x, out, "outcomes", 32)
        print(f"elapsed time: {time() - st:.2f}")
        self.assertTrue(np.array_equal(out[:, 0, 0, 0], np.where(x[:, 1] <= 0, 1, 2)))


if __name__ == "__main__":
    ut.main()
