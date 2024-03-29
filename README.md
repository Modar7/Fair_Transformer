# Fair Classification via Transformer Neural Networks: Case Study of an Educational Domain.


Original Implementation of the paper [[link]](https://arxiv.org/abs/2206.01410): [Fair Classification via Transformer Neural Networks: Case Study of an Educational Domain](Fair_Transformer.pdf) by Modar Sulaiman, Kallol Roy.


# Usage
* In order to reproduce the results, please install the required packages using the following command: `conda env create -f environment.yml`

# Running experiments

* `fair_trainer.py` for training the model with fairness constraint (avoiding the disparate treatment) in the main objective function based on the following two references:
   * [Fairness Constraints: A Flexible Approach for Fair Classification](https://jmlr.org/papers/v20/18-262.html)
   * [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)

* `trainer.py` for training the model without any fairness constraint in the main objective function.

*  In each of the previous trainers (`fair_trainer.py` and `trainer.py`) you can choose one of the follwoing transformer-based models for training: [Tab-Transformer](https://arxiv.org/abs/2012.06678), [FT-Transformer](https://arxiv.org/pdf/2106.11959.pdf), [SAINT](https://arxiv.org/abs/2106.01342) and [Perceiver](https://arxiv.org/abs/2103.03206).

* `testing.py` for evaluating the trained model on the test dataset.



# License & copyright
Licensed under the [MIT License](License).


