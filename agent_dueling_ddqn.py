import os
import tensorflow as tf
import math
import pandas as pd
import numpy as np
import random
from sklearn import metrics

SEED = 11037
random.seed(SEED)
tf.set_random_seed(SEED)

tf_config = tf.compat.v1.ConfigProto()

class Agent:
    def __init__(self, data, config):
    	#initialize everything here
        self.data = data
        self.config = config
        self.memory = self.ReplayMemory
        self.qnetwork = self.DuelDDQN(self.config)
        self.total_classes = self.qnetwork.total_classes
        self.q_eval = self.qnetwork.q_curr_net
        self.q_target = self.qnetwork.q_target_net
        self.epsilon_start = self.config.epsilon_start
        self.epsilon_end = self.config.epsilon_end
        self.minority_classes = self.config.minority_classes
        self.epsilon = 1
        self.epsilon_decay_steps = 120000

        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(config=tf_config)
    
    def check_filename(self,file_name):
        if os.path.isfile(file_name):
            expand = 1
            while True:
                expand += 1
                new_file_name = file_name.split(".csv")[0] + str(expand) + ".csv"
                if os.path.isfile(new_file_name):
                    continue
                else:
                    file_name = new_file_name
                    break
            return file_name
        else:
            return file_name

    # calculate the softmax of a vector
    def softmax(self, vector):
        e = np.exp(vector)
        return e / e.sum()

    def scale(self, A):
        return (A-np.min(A))/(np.max(A) - np.min(A))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = [np.random.randint(self.total_classes)]
            raw_action = action
        else:
            q = self.sess.run(self.q_eval, feed_dict={self.qnetwork.state: state})
            action = np.argmax(q, axis=1)
            r_softmax = self.softmax(q)
            raw_action = r_softmax[:,1]
        return action, raw_action

    def get_reward(self, label, action, z):
        terminate_training = 0
        if action == label:
            reward = self.data.reward_weights_z[int(z)]
        else:
            reward = - self.data.reward_weights[int(label)]
            # get terminate_training = 1 if minority class misclassified
            if label in self.minority_classes:
                terminate_training = 1
        return [reward], [terminate_training]

    def update_epsilon(self, training_steps):
        self.epsilon = np.clip(
            (self.epsilon_end - self.epsilon_start) / self.epsilon_decay_steps * training_steps + 1,
            *[self.epsilon_end, self.epsilon_start])
    
    def current_progress(self, training_label, train_prediction, val_label, val_prediction, pred_raw, step, current_dataset="Train"):
        train_prediction = np.concatenate(train_prediction)
        print("training_steps : {}".format(step))
        if current_dataset == "Train":
            labels = [training_label]
            predictions = [train_prediction]
        else:
            labels = [val_label]
            predictions = [val_prediction]
        for idx, (label, prediction) in enumerate(zip(labels, predictions)):
            recall, specificity, recall_score, specificity_score, ppv_score, npv_score, auc_score, conf_l, conf_u = calc_metrics(label, prediction, pred_raw)
            print("\n")
        return recall, specificity, recall_score, specificity_score, ppv_score, npv_score, auc_score, conf_l, conf_u            

    def train(self):
        metric_names = ['recall','specificity','ppv','npv','auroc','conf_l','conf_u','q_target','q_est']
        results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
        results_df_v = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
        results_df_t = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(self.qnetwork.update_target, feed_dict={self.qnetwork.target_update_rate: 1.})
        print("start training")
        eval_s_list = np.array_split(self.data.x_valid, 23132)
        eval_y_list = list(self.data.y_test)
        eval_finals_list = np.array_split(self.data.x_test_final, 22857) 
        eval_finaly_list = list(self.data.y_test_final)

        at = np.random.randint(self.data.x_train.shape[0])
        s = self.data.x_train[at]  
        y = self.data.y_train[at]  
        training_steps = 0
        # used for threshold adjustment
        pred_list, label_list, pred_list_raw = [], [], []  
        current_highest_val = 0
        num_it = 0
        highest_training_steps = 0
        while training_steps < self.config.training_steps:
            a, a_raw = self.get_action(s[np.newaxis, ...])  
            r, t = self.get_reward(y, a) 
            at_ = np.random.randint(self.data.x_train.shape[0])
            s_ = self.data.x_train[at_] 
            y_ = self.data.y_train[at_]  
            self.memory.addsample([s, a, r, s_, t])

            if (len(self.buffer) >= 5000):
                sample_s, sample_a, sample_r, sample_s_, sample_t = self.memory.sample_tuple(1)
                q_eval, q_target = self.sess.run([self.q_eval, self.q_target],
                                               feed_dict={self.qnetwork.state: sample_s_})

                q_eval_a = np.argmax(q_eval, axis=1)[:, np.newaxis]  
                max_q_ = np.take_along_axis(q_target, q_eval_a, axis=1)  
                self.sess.run(self.qnetwork.opt, feed_dict={self.qnetwork.state: sample_s, self.qnetwork.action: sample_a,
                                                            self.qnetwork.reward: sample_r, self.qnetwork.terminate_training: sample_t,
                                                            self.qnetwork.target_q: max_q_,
                                                            self.qnetwork.learning_rate: self.config.learning_rate})
                
                pred_list.append(a)
                pred_list_raw.append(a_raw)
                label_list.append(y)
                training_steps += 1
                self.update_epsilon(training_steps)
                
                if training_steps % self.config.target_update_step == 0:
                    self.sess.run(self.qnetwork.update_target, feed_dict={self.qnetwork.target_update_rate: self.config.target_update_rate})

                if training_steps % self.config.show_progress == 0:
                    # validation data
                    eval_a_list = []
                    eval_a_raw_list = []
                    for eval_s in eval_s_list:
                        eval_a, eval_a_raw = self.get_action(eval_s, False)  
                        eval_a_list.extend(list(eval_a))
                        eval_a_raw_list.extend(list(eval_a_raw))
                    
                    # test data
                    eval_finala_list = []
                    eval_finala_raw_list = []
                    for eval_finals in eval_finals_list:
                        eval_finala, eval_finala_raw = self.get_action(eval_finals, False)#take greedy policy for val data.
                        eval_finala_list.extend(list(eval_finala))
                        eval_finala_raw_list.extend(list(eval_finala_raw))
                        
                    print("Train Scores")
                    recall, specificity, recall_test, specificity_test, ppv_test, npv_test, auc_score, conf_l, conf_u = self.current_progress(label_list, pred_list, eval_y_list, eval_a_list, pred_list_raw, training_steps, current_dataset = 'Train')
                    results_df = results_df.append({'recall': recall_test, 'specificity':specificity_test, 'ppv':ppv_test, 'npv':npv_test, 'auroc':auc_score, 'conf_l':conf_l, 'conf_u':conf_u, 'q_target':q_target,'q_est':value_est}, ignore_index = True)
                    
                    print("Validation Scores")
                    recall_v, specificity_v, recall_test_v, specificity_test_v, ppv_test_v, npv_test_v, auc_score_v, conf_l_v, conf_u_v = self.current_progress(label_list, pred_list, eval_y_list, eval_a_list, eval_a_raw_list, training_steps, current_dataset = 'Validation')
                    results_df_v = results_df_v.append({'recall': recall_test_v, 'specificity':specificity_test_v, 'ppv':ppv_test_v, 'npv':npv_test_v, 'auroc':auc_score_v, 'conf_l':conf_l_v, 'conf_u':conf_u_v, 'q_target':q_target,'q_est':value_est}, ignore_index = True)

                    
                    results_df.to_csv('auc_scores.csv')
                    results_df_v.to_csv('auc_scores_v.csv')
                    raw_results_v = pd.DataFrame(eval_a_raw_list)
                    raw_results_v.to_csv("results_val_rawval" + str(self.imbalance_level) +".csv", index = False)

                    #monitor for early stopping
                    if ((recall_v > self.config.sensitivity_es) & (specificity_v > self.config.specificity_es)) or (training_steps == self.config.end_training):

                        metric_names = ['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC']
                        thresh_adj_results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
                        best_threshold = 0
                        metric_of_interest = 'Recall'
                        desired_metric_value = self.config.threshold_opt
                        match_number = 20
                        error = self.config.threshold_opt_range
                        
                        print("Test Scores")
                        recall_t, specificity_t, recall_test_t, specificity_test_t, ppv_test_t, npv_test_t, auc_score_t, conf_l_t, conf_u_t = self.current_progress(label_list, pred_list, eval_finaly_list, eval_finala_list, eval_finala_raw_list, training_steps, current_dataset = 'Validation')
                        results_df_t = results_df_t.append({'recall': recall_test_t, 'specificity':specificity_test_t, 'ppv':ppv_test_t, 'npv':npv_test_t, 'auroc':auc_score_t, 'conf_l':conf_l_t, 'conf_u':conf_u_t}, ignore_index = True)
                    
                        results_df_t.to_csv('auc_scores_ouh_t' + str(self.imbalance_level) +'.csv')
                        raw_results_t = pd.DataFrame(eval_finala_raw_list)
                        raw_results_t.to_csv("results_ouh_t_rawval" + str(self.imbalance_level) +".csv", index = False)
                        
                        label_list.clear(), pred_list.clear(), pred_list_raw.clear()
                        
                        #get threshold from validation set
                        best_threshold, thresh_adj_results_df = get_threshold(eval_a_raw_list, eval_y_list, metric_of_interest, desired_metric_value, error, match_number)
                        print(best_threshold)
                        print(thresh_adj_results_df)
                        print('\n')
                        print("Threshold Adjusted Performance Metrics:")
                        n1 = len(pd.Series(eval_finaly_list)[pd.Series(eval_finaly_list) == 1])
                        n2 = len(pd.Series(eval_finaly_list)[pd.Series(eval_finaly_list) == 0])
                        accuracy_test, recall_test, precision_test, specificity_test, f1score_test, ppv_test, npv_test, roc_auc_test = get_test_results_from_threshold(eval_finala_raw_list, eval_finaly_list, best_threshold,n1,n2)
                                                
                        self.saver.save(self.sess, './model/model-trainingstep')
                        os.exit()
                    else:
                        label_list.clear(), pred_list.clear(), pred_list_raw.clear()

            at, s, y = at_, s_, y_


class ReplayMemory:
    def __init__(self):
        self.next = col.namedtuple("next", ['s', 'a', 'r', 's_', 't'])
        self.buffer = []

    def addsample(self, sample):
        if len(self.buffer) >= 50000:
            del self.buffer[0]
        self.buffer.append(self.next(*sample))

    def sample_tuple(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        s = np.array([var.s for var in sample])  
        a = np.array([var.a for var in sample])  
        r = np.array([var.r for var in sample]) 
        s_ = np.array([var.s_ for var in sample])  
        t = np.array([var.t for var in sample])  
        return s, a, r, s_, t


class DuelDDQN:
    def __init__(self, config):
        n_feat = 25
        self.total_classes = 2
        self.config = config

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        with tf.compat.v1.variable_scope('input'):
            self.learning_rate = tf.compat.v1.placeholder(dtype=tf.float32)
            self.state = tf.compat.v1.placeholder(shape=[1,n_feat], dtype=tf.float32)
            self.action = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.int32)
            self.reward = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.terminate_training = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.int32)
            self.target_q = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.target_update_rate = tf.compat.v1.placeholder(dtype=tf.float32)

        with tf.compat.v1.variable_scope('target_net'):
            self.q_target_net = self.get_q_network()

        with tf.compat.v1.variable_scope('current_net'):
            self.q_curr_net = self.get_q_network()

        self.update_target = [t.assign((1 - self.target_update_rate) * t + self.target_update_rate * c) 
        for t, c in zip(tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="target_net"), 
            tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="current_net"))]
        self.q_curr_a = tf.expand_dims(tf.gather_nd(self.q_curr_net, self.action, batch_dims=1), axis=1)
        self.target = self.reward + (1 - self.terminate_training) * self.config.gamma * self.target_q
        self.loss = tf.losses.huber(self.target, self.q_curr_a)
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, 
            var_list=tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="current_net"))

    def get_q_network(self):
        x = tf.compat.v1.layers.dense(self.state, 100, activation=tf.nn.relu)
        x = tf.compat.v1.nn.dropout(x, rate = 0.3)
        a = tf.compat.v1.layers.dense(x, self.total_classes)
        v = tf.compat.v1.layers.dense(x, 1)
        q = v + a - tf.compat.v1.nn.softmax(a, axis=1)
        return q