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

import numpy as np

from openasce.discovery import CausalSearchDiscovery


def main():
    cs = CausalSearchDiscovery()
    cs.fit(X=np.loadtxt("search_samples.csv", delimiter=",", dtype=int))
    (g, s) = cs.get_result()
    print(f"score={s}")
    edges = [(p, c) for c, y in g.parents.items() for p in y]
    print(f"edge num={len(edges)}")


if __name__ == "__main__":
    main()
