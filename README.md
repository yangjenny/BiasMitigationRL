# Reinforcement Learning for Bias Mitigation

This repository hosts the version of the code used for the publication ["Algorithmic Fairness and Bias Mitigation for Clinical Machine Learning: A New Utility for Deep Reinforcement Learning"](https://www.medrxiv.org/content/10.1101/2022.06.24.22276853v1). 

## Dependencies

We have tested this implementation using Python version 3.6.9 and Tensorflow version 2.6.2 on a linux OS machine. To use this branch, you can run the following lines of code:

```
git clone https://github.com/yangjenny/BiasMitigationRL.git
cd BiasMitigationRL
pip install -e .
```

## Getting Started

To run code: 

```
python BiasMitigationRL/run.py
```

(UCI Adult dataset automatically loaded for training)

This example uses the UCI Adult dataset, where one is trying to classify income (two classes: <=50K and >50K). Additional details about the dataset, including all attributes included, can be found [here](https://archive.ics.uci.edu/ml/datasets/Adult).

## Citation

If you found our work useful, please consider citing:

Yang, J., Soltan, A. A., & Clifton, D. A. (2022). Algorithmic Fairness and Bias Mitigation for Clinical Machine Learning: A New Utility for Deep Reinforcement Learning. medRxiv.


