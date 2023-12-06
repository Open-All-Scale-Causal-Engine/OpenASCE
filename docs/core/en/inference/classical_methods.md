# The Classical Methods of Causal Inference

We have introduced and adapted some commonly used methods of causal inference from [Econml](https://econml.azurewebsites.net/spec/estimation.html), which can be well suited for task scheduling and execution in this platform.

## Meta-Learners
Meta-Learners models the response target and estimates the treatment effect by quantifying the change in the target variable caused by the treatment.
It contains three main methods: T-Learner, S-Learner and X-Learner.

The example: `examples/openasce/inference/learner/metalearners.py`

## Doubly Robust Learning
DRLearner is a doubly robust estimation method based on two-stage estimation that can effectively estimate heterogeneity when there are confounders in the observed data, which can reduce bias effectively.

The example: `examples/openasce/inference/learner/drlearner.py`

## Double Machine Learning
DML is a method used in studying Heterogeneous Treatment Effects (HTE) that allows for unbiased estimation of Average Treatment Effects (ATE) even when the estimation of the nuisance parameter W is biased, by estimating the moment residuals.

The example: `examples/openasce/inference/learner/dml.py`