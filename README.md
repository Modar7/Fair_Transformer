# Fair Classification via Transformer Neural Networks: Case Study of an Educational Domain.


Original Implementation of the paper: [Fair Classification via Transformer Neural Networks: Case Study of an Educational Domain] by Modar Sulaiman, Kallol Roy.



# License & copyright
Licensed under the [MIT License](License).


# Usage
I onrder to reproduce the results, please install the required packages using the following command: 

conda env create -f environment.yml

# Running experiments

* You can train a transfomer model with the fairness constraint in the main objective function (for avoiding the disparate treatment) using the following: python fair_trainer.py

* You can train transfomer model without any fairness constraint in the main objective function using the following: python fair_trainer.py


