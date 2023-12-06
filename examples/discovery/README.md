# Causal Discovery

In this directory, we provide two examples to discover the causal relations using sample data. They can be run via `python {filename}` .

`search-discovery.py` uses the score-based approach to learn the structure of the causal graph, and the greedy search method ( `K2` algorithm) is used to explore the nodes. The search algorithm first assumes a topological order of the graph and restricts the variable's parent set to the variables with a higher order. It takes a greedy approach by adding the parent that increases the score most until no improvement can be made.

`regression-discovery.py` uses the other way to learn the structure. It formulates the structure learning problem as a purely continuous optimization problem over real matrices so that avoids the combinatorial constraint entirely, referring to [link](https://arxiv.org/abs/1803.01422) and [link](https://arxiv.org/abs/1909.13189).
