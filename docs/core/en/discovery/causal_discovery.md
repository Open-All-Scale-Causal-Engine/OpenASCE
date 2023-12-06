# Causal Discovery

To get the causal relations from the samples, we can search and optimize the structure of the structure of the causal graph. The following method is used to achieve the goal.

## Search Discovery

The search discovery uses the score-based approach to learn the structure of the causal graph, and the greedy search method ( `K2` algorithm) is used to explore the nodes. The search algorithm first assumes a topological order of the graph and restricts the variable's parent set to the variables with a higher order. It takes a greedy approach by adding the parent that increases the score most until no improvement can be made. Refer to the example `examples/openasce/discovery/search-discovery.py` to see the details.

## Regression Discovery

The regression discovery uses the other way to learn the structure. It formulates the structure learning problem as a purely continuous optimization problem over real matrices so that avoids the combinatorial constraint entirely, referring to [link](https://arxiv.org/abs/1803.01422) and [link](https://arxiv.org/abs/1909.13189). The example `examples/openasce/discovery/regression-discovery.py` shows the usage.

## References

[1] Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). [DAGs with NO TEARS: Continuous optimization for structure learning](https://arxiv.org/abs/1803.01422)
([NeurIPS 2018](https://nips.cc/Conferences/2018), Spotlight)

[2] Zheng, X., Dan, C., Aragam, B., Ravikumar, P., & Xing, E. P. (2020). [Learning sparse nonparametric DAGs](https://arxiv.org/abs/1909.13189) ([AISTATS 2020](https://aistats.org/), to appear).

[3] Marco Scutari. (2018). Dirichlet Bayesian Network Scores and the Maximum Relative Entropy Principle.

[4] Puhani Patrick. (2000). The Heckman correction for sample selection and its critique.

[5] Shahab Behjati, Hamid Beigy. (2020). [Improved K2 algorithm for Bayesian network structure learning.](https://www.sciencedirect.com/science/article/pii/S095219762030083X)
