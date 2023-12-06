# Causal Debiasing Algorithms

## Introduction

Causal debias learning mainly utilizes human a priori to help neural network models remove data biases, such as polularity bias and selection bias, which are common in recommendation scenarios [1].
Being affected by data bias, model training is also biased, and we adjust the observed data with the help of causal means to reduce the impact of bias. These methods can also be used to estimate causal effects like end-to-end T-model, Doubly robust learning. The difference with classical causal methods is that the classical implementation is multi-stage (containing multiple training, prediction, and sample organization stages), whereas neural network (NN)-based implementations are end-to-end, and thus more efficient.

The openasce precipitates the following debias methods:

| Methods | Brief | Demo |
| -----   | ----  | ---- |
| CFR[2]  | One branch of each experimental and control group is de-fitted, adding distance constraints on both sides of the representation.  | /examples/openasce/debias/cfr |
| DICE[3] | A multitasking framework that disentangle training representations into interest representations and conformity representations. | /examples/openasce/debias/dice |
|DMBR[4]|Correction of observational data with the help of causal diagrams and backdoor adjustments to remove the effects of matching bias| /examples/openasce/debias/dmbr|
|FAIRCO[5]|Debiased dynamic learning-to-rank problem for scenarios with rich-get-richer (Matthew effect) and exposure bias (active exposure based on value or strategy, fairness).|/examples/openasce/debias/fairco|
|IPW[6]|Correcting data distributions by inverse probability weighting.|/examples/openasce/debias/ipw|
|MACR[7]|Removing item's popularity bias and conformity bias through multi-task learning.|/examples/openasce/debias/macr|
|PDA[8]|Observations are corrected with the help of causal diagrams and backdoor adjustments to remove the effects of popularity bias, and the final output scores are adjusted by the scale of the item's table exposure.|/examples/openasce/debias/pda|
|Doubly Robust[9]|In recommender systems, usually the ratings of a user to most items are missing and a critical problem is that the missing ratings are often missing not at random (MNAR) in reality. A double robust method is used to correct the bias of this non-random observation sample.|/examples/openasce/debias_drobust|
|IPS[10]|The inverse propensity score method (IPS) weights the samples by propensity score to reduce data bias. The implemented ips method is based on neural network prediction of propensity scores and supports multiple treament.|/examples/openasce/debias/debias_ips|


## Demonstration of Usage

The related dmeo are all in the openasce code base /examples/openasce/debias directory. The following is an example of how to use MACR.

MACR (Model-Agnostic Counterfactual Reasoning): Through the framework of multi-tasking, the popularity of the item and the conformity of the user are modeled as auxiliary tasks, and these two tendencies are subtracted in the inference phase to obtain an unbiased user-item matching score. user-item matching score.

### Prepare Data

Provide user features and item features separately, and pass them to the model in a dict, like {'user': user, 'item': item}.
### Params Setting

Parameters are divided into default parameters and user-defined parameters. Default parameters are not recommended to be modified if you cannot understand their meaning. The user-defined parameters are the colum of the data and the path of the data.

### Run code

Execute the command "python macr_data_test.py".



[1]: [Bias and Debias in Recommender System: A Survey and Future Directions.](https://arxiv.org/pdf/2010.03240.pdf)

[2]: [Estimating individual treatment effect: generalization bounds and algorithms.](http://proceedings.mlr.press/v70/shalit17a/shalit17a.pdf)

[3]: [Disentangling Interest and Conformity with Causal Embedding.](https://arxiv.org/pdf/2006.11011.pdf)

[4]: [Alleviating Matching Bias in Marketing Recommendations.](https://dl.acm.org/doi/abs/10.1145/3539618.3591854)

[5]: [Controlling Fairness and Bias in Dynamic Learning-to-Rank.](https://arxiv.org/pdf/2005.14713.pdf.)

[6]: [Inverse probability weighted estimation for general missing data problems.](https://www.econstor.eu/bitstream/10419/79298/1/386079048.pdf)

[7]: [Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system.](https://arxiv.org/pdf/2010.15363.pdf)

[8]: [Popularity-bias Deconfounding and Adjusting.](https://arxiv.org/pdf/2105.06067.pdf)

[9]: [Doubly robust joint learning for recommendation on data missing not at random.](https://proceedings.mlr.press/v97/wang19n.html)

[10]: [Estimating Causal Effects from Large Data Sets Using Propensity Scores.](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=15bff7ea2499425f73bbaa8bb9382b1dd84f2782)
