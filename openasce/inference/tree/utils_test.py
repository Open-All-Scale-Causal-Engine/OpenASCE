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

from time import time
import unittest as ut

import numpy as np
import pandas as pd

# from .utils import *
from openasce.inference.tree.utils import *


class TestAggrate(ut.TestCase):
    def test_update_x_map(self):
        n, m, l, n_y, n_bins = 100, 10, 10, 2, 10
        x = np.random.randint(0, n_bins, size=(n, m), dtype=np.int32)
        index = np.random.permutation(n).astype(np.int32)
        split_infos = np.concatenate(
            [
                np.random.randint(0, m, size=(l, 1), dtype=np.int32),
                np.random.randint(0, n_bins - 1, size=(l, 1), dtype=np.int32),
            ],
            axis=1,
        )
        tmp = np.sort(np.random.choice(range(n), size=(l + 1,), replace=False))
        leaves_range = np.stack([tmp[:-1], tmp[1:]], axis=1).astype(np.int32)
        out = np.zeros([l * 2, 2], np.int32)
        update_x_map(x, index, split_infos, leaves_range, out)
        # check split
        for i in range(l):
            self.assertEqual(out[i * 2, 0], leaves_range[i, 0])
            self.assertEqual(out[i * 2 + 1, 1], leaves_range[i, 1])
        for i in range(l):
            fid, thresh = split_infos[i]
            fr, end = out[i * 2]
            for j in range(fr, end):
                self.assertLessEqual(x[index[j], fid], thresh)
            fr, end = out[i * 2 + 1]
            for j in range(fr, end):
                self.assertGreater(x[index[j], fid], thresh)

    def test_update_histogram(self):
        n, m, l, n_y, n_w, n_bins = 100, 100, 10, 2, 2, 10
        x = np.random.randint(0, n_bins, size=(n, m), dtype=np.int32)
        y = np.random.normal(size=(n, n_y))
        w = np.random.randint(0, n_w, size=(n,), dtype=np.int32)
        index = np.random.permutation(n).astype(np.int32)
        tmp = (
            [0]
            + list(np.sort(np.random.choice(range(n), size=(l - 1,), replace=False)))
            + [n]
        )
        leaves_range = np.stack([tmp[:-1], tmp[1:]], axis=1).astype(np.int32)
        out = np.zeros([l, m, n_bins, n_w, n_y], y.dtype)
        print("*" * 50)
        time_start = time()
        update_histogram(y, x, index, leaves_range, w, out, range(l), n_w, n_bins, 32)
        print(f"elapsed time: {time() - time_start:.3f}s")
        print("*" * 50)
        df_y = pd.DataFrame(y)
        for f in range(m):
            for leaf in range(l):
                fr, end = leaves_range[leaf]
                idx = index[fr:end]
                tmp = df_y.loc[idx].groupby([x[idx, f], w[idx]]).sum()
                for i in range(n_bins):
                    for j in range(n_w):
                        if (i, j) in tmp.index:
                            self.assertTrue(
                                np.abs(out[leaf, f, i, j] - tmp.loc[i, j]).mean() < 1e-6
                            )
                        else:
                            self.assertEqual(np.abs(out[leaf, f, i, j]).sum(), 0)

    def test_indexbyarray(self):
        n = 10000
        n_treatment = 2
        n_outcome = 8
        y = np.random.normal(0, 1, size=(n, n_treatment, n_outcome))
        by = np.random.randint(0, n_treatment, size=(n), dtype=np.int32)
        o1 = np.zeros([n, n_outcome], dtype=y.dtype)
        o2 = np.zeros([n, n_outcome], dtype=y.dtype)
        indexbyarray(y, by, o1, o2)
        indexbyarray(y, by, o1)
        for i in range(n):
            self.assertTrue(np.array_equal(y[i, by[i]], o1[i]))
            self.assertTrue(np.array_equal(y[i, 1 - by[i]], o2[i]))


if __name__ == "__main__":
    ut.main()
