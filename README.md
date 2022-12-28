# Reinforcement Learning for Bias Mitigation

This repository hosts the version of the code used for the publication ["Algorithmic Fairness and Bias Mitigation for Clinical Machine Learning: A New Utility for Deep Reinforcement Learning"](https://www.medrxiv.org/content/10.1101/2022.06.24.22276853v1). 

## Dependencies

We have tested this implementation using Python version 3.6.9 and Tensorflow version 2.6.2 on a linux OS machine. To use this branch, you can run the following lines of code:

```
conda create -n BiasMitigationEnv python==3.7
conda activate BiasMitigationEnv
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

This example uses the UCI Adult dataset, where one is trying to classify income (two classes: <=50K and >50K), and mitigate gender (male vs female) bias. Additional details about the dataset, including all attributes included, can be found [here](https://archive.ics.uci.edu/ml/datasets/Adult).

After training, performance metrics (auroc,npv,ppv,recall,specificity) and raw prediction results will be saved as csv files in the path. 
An example run and expected output can be found in BiasMitigationRL/example/training_example.ipynb

## Citation

If you found our work useful, please consider citing:

Yang, J., Soltan, A. A., & Clifton, D. A. (2022). Algorithmic Fairness and Bias Mitigation for Clinical Machine Learning: A New Utility for Deep Reinforcement Learning. medRxiv.


