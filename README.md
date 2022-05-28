# Fair Classification via Transformer Neural Networks: Case Study of an Educational Domain.


Original Implementation of the paper: [Fair Classification via Transformer Neural Networks: Case Study of an Educational Domain] by Modar Sulaiman, Kallol Roy.



# License & copyright
Licensed under the [MIT License](License).


# Usage
* I order to reproduce the results, please install the required packages using the following command: 

                        conda env create -f environment.yml

# Running experiments

* `fair_trainer.py` for training the model with the fairness constraint (avoiding the disparate treatment) in the main objective function.

* `trainer.py` for training the model withput any fairness constraint in the main objective function.

*  Yuo can train one of the follwoing transformer-based models: Tab-Transformer, FT-Transformer, SAINT and [Perceiver](https://arxiv.org/abs/2103.03206).

