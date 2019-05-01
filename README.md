# Similarity Learning for Pointwise ROC Optimization


## About
The contents of this repository cover the experiments described in the paper:

[*A Probabilistic Theory of Supervised Similarity Learning for Pointwise ROC Curve Optimization.*
Robin Vogel, Aurélien Bellet, Stéphan Clémençon ; Proceedings of the 35th International Conference on Machine Learning, PMLR 80:5062-5071, 2018.](http://proceedings.mlr.press/v80/vogel18a.html)

Three experiments are described in the paper:
* Experiment 1: **Pointwise ROC optimization** - Describes how we can optimize for a specific point of the ROC curve, with guarantees derived from our analysis, in a very simple special case.
* Experiment 2: **Fast Rates** - Shows that the fast rates can be illustrated with simples distributions, when we satisfy the assumptions of a Mammen-Tsybakov type assumption.
* Experiment 3: **Scalability by sampling** - Shows that for the [MMC algorithm](https://dl.acm.org/citation.cfm?id=2968618.2968683), which is a metric learning objective which formulation is very close to our problem, subsampling very agressively the negative pairs does not hinder learning.

## Required libraries

* Experiment 1: numpy, matplotlib.
* Experiment 2: numpy, matplotlib, pandas, scipy.stats
* Experiment 3: numpy, matplotlib, scikit-learn, autograd, configargparse.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

### [Robin Vogel](https://perso.telecom-paristech.fr/rvogel/), 2018.
