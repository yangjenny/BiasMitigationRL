# Reinforcement Learning for Bias Mitigation

This repository hosts the version of the code used for the publication ["Algorithmic fairness and bias mitigation for clinical machine learning with deep reinforcement learning"](https://www.nature.com/articles/s42256-023-00697-3). 

## Dependencies

We have tested this implementation using:
1. Python version 3.6.9 and Tensorflow version 2.6.2 on a linux OS machine. 
2. Python version 3.9.2 and Tensorflow version 2.11.0 on a mac OS machine (Big Sur). 

To use this branch, you can run the following lines of code:

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
An example run and expected output can be found in example/training_example.ipynb

## Citation

If you found our work useful, please consider citing:

Yang, J., Soltan, A. A., Eyre, D. W., & Clifton, D. A. (2023). Algorithmic fairness and bias mitigation for clinical machine learning with deep reinforcement learning. Nature Machine Intelligence, 1-11.


