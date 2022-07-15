import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import random
import os


class Data:
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.get_rewards()
        self.get_rewards_z()

    def load_data(self):
        age_match = self.config.age_match
        testfile = 'puh'
        print("Test file: " + testfile)
        x_train = pd.read_pickle('data/X_train.pkl')
        y_train = pd.read_pickle('data/y_train.pkl')
        x_valid= pd.read_pickle('data/X_valid.pkl')
        y_valid= pd.read_pickle('data/y_valid.pkl')
        x_test_final = pd.read_pickle('data/' + testfile + '_test/X_test.pkl')
        y_test_final = pd.read_pickle('data/' + testfile + '_test/y_test.pkl')
        z_train = pd.read_pickle('data//Z_ethnicity_train.pkl')
        z_valid = pd.read_pickle('data//Z_ethnicity_valid.pkl')
        z_test_final = pd.read_pickle('data/' + testfile + '_test/Z_ethnicity_test.pkl')
        
        if age_match:
            get_new_control_indices = False
            
            Xtrain=x_train
            ytrain=y_train
            zage = z_age_train
            matched_cohort_indices = []
            match_number = self.config.imbalance_level
            idx_control = [i for i in range(len(ytrain)) if ytrain[i] == 0]
            control_data = Xtrain.iloc[idx_control,:]
            print('original xtrain control: ', control_data.shape)
            control_y = [ytrain[i] for i in idx_control]
            #control_z = [ztrain[i] for i in idx_control]
            control_age = [zage[i] for i in idx_control]
            idx_case = [i for i in range(len(ytrain)) if ytrain[i] == 1]
            case_data = Xtrain.iloc[idx_case,:]
            case_y = [ytrain[i] for i in idx_case]
            #case_z = [ztrain[i] for i in idx_case]
            case_age = [zage[i] for i in idx_case]
            if get_new_control_indices == True:
                count = 1
                for index in idx_case:
                    print(str(count))
                    patient_data = Xtrain.iloc[index,:]
                    patient_age = pd.Series(zage).iloc[index]#.numpy()
                    #patient_z = z_ethnicity[index].numpy()
                    age_condition = control_age == patient_age
                    #z_condition = control_z == patient_z
                    matched_indices_bool = age_condition #& z_condition
                    matched_indices= np.array(idx_control)[matched_indices_bool]
                    random.seed(0)
                    random.shuffle(matched_indices)
                    valid_indices = list(set(matched_indices)-set(matched_cohort_indices))[:match_number]
                    #matched_indices = random.sample(matched_indices, len(matched_indices))
                    #valid_indices = [index for index in matched_indices if index not in matched_cohort_indices][:match_number]
                    matched_cohort_indices.extend(valid_indices)
                    count=count+1
                #print('TRUE: ', matched_cohort_indices)
                with open(os.path.join('control_indices_%i.pkl' % (match_number)),'wb') as f:
                            pickle.dump(matched_cohort_indices,f)
            else:
                with open(os.path.join('control_indices_%i.pkl' % (match_number)),'rb') as f:
                    matched_cohort_indices = pickle.load(f)
                #print('FALSE: ', matched_cohort_indices)
            control_matched_data = Xtrain.iloc[matched_cohort_indices,:]
            print('new xtrain matched control: ', control_matched_data.shape)
            #control_matched_z = [ztrain[i] for i in matched_cohort_indices]
            control_matched_y = [ytrain[i] for i in matched_cohort_indices]
            x_train = np.concatenate((control_matched_data, case_data), axis=0)
            #print(Xtrain.shape)
            y_train = np.concatenate((control_matched_y + case_y),axis=None)

        self.y_train, self.y_valid= y_train, y_valid
        self.x_train, self.x_valid= x_train, x_valid
        self.x_test_final, self.y_test_final = x_test_final, y_test_final

    def get_rewards(self):
        class_numbers = np.unique(self.y_train, return_counts=True)[1]
        reward_values = 1 / class_numbers#**(1/2)
        self.reward_weights = np.round(reward_values / np.linalg.norm(reward_values), 4)

    def get_rewards_z(self):
        class_numbers = np.unique(self.z_train, return_counts=True)[1]
        reward_values = 1 / class_numbers#**(1/2)
        self.reward_weights_z = np.round(reward_values / np.linalg.norm(reward_values), 4)

