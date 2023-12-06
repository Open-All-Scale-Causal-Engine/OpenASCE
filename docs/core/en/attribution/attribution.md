# Attribution

The causal attribution tries to find out why events and behaviors occur. Through the attribution, we can classify the features and their values that have more impact under the treatment by a set of descriptive rules. The general combinatorial optimization problem is NP-hard, so the search procedure with multiple iterations is used for attribution and the branch-and-bound method is used to improve its efficiency. The search method will generate, evaluate, and keep the best conditions (node and value pair) satisfying the pre-defined probability threshold in each iteration, and terminate when the iteration number exceeds the required maximal rule length.

The example `examples/openasce/attribution/attribution.py` shows the usage and result through the causal graph and samples.
