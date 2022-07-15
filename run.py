import argparse
import ast
from config import config
from data import Data
from agent_dueling_ddqn import Agent

def parse():
    parser = argparse.ArgumentParser()

    #Load data
    parser.add_argument('--minority_classes', type=ast.literal_eval, default=[1],
                        help='Classes to be used as minor classes among the newly changed classes')
    parser.add_argument('--age_match', type=bool, default=True, help='Configures age matching for matched case:controls')
    parser.add_argument('--imbalance_level', type=int, default=20, help='Set majority to minority instance')

    #Training config
    parser.add_argument('--training_steps', type=int, default=120000, help='Total training steps used')
    parser.add_argument('--learning_rate', type=float, default=0.0004, help='Learning rate')
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--target_update_rate', type=float, default=1., help='Rate of updating the target q network')
    parser.add_argument('--target_update_step', type=int, default=6000, help='Period to update the target q network')
    parser.add_argument('--end_training', type=int, default=120000, help='Steps to end training and save model')
    parser.add_argument('--show_progress', type=int, default=6000, help='Steps to check training progress')

    #Thresholding and Optimization
    parser.add_argument('--threshold_opt', type=float, default=0.9, help='Sensitivity to optimize results to')
    parser.add_argument('--threshold_opt_range', type=float, default=0.02, help='Range for thresholding grid search')
    parser.add_argument('--sensitivity_es', type=float, default=0.85, help='Monitoring sensitivity of validation set for early stop')
    parser.add_argument('--specificity_es', type=float, default=0.75, help='Monitoring specificity of validation set for early stop')

    config = parser.parse_args()
    return config



if __name__ == '__main__':
    data = Data(parse())
    agent = Agent( data, parse())
    agent.train()