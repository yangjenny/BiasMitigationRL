from sklearn import metrics
import math
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score
import pandas as pd


def get_threshold(yprob, ytrue, metric_of_interest, desired_metric_value, error, match_number):
    metric_names = ['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC']
    thresh_adj_results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
    best_threshold = 0
    y = pd.Series(ytrue)
    probs = pd.Series(yprob)
    threshold_metrics = pd.DataFrame(np.zeros((1000,8)),index=np.linspace(0,1,1000),columns=['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC'])
    prev = 1/(match_number+1)
    for t in np.linspace(0,1,1000):
        pred = np.where(probs>t,1,0)
        pred = pd.Series(pred)
        true_neg, false_neg, false_pos, true_pos = confusion_matrix(pred, y)
        accuracy = accuracy_score(pred, y)
        recall =  true_pos / (false_neg + true_pos)
        precision = true_pos / (false_pos + true_pos)
        specificity = true_neg/(true_neg+false_pos)
        #for ppv and npv, set prevalance 
        prev = 1/(match_number+1)
        ppv = (recall* (prev))/(recall * prev + (1-specificity) * (1-prev))
        if true_neg== 0 and false_neg==0:
            npv = 0
        else:
            npv = (specificity* (1-prev))/(specificity * (1-prev) + (1-recall) * (prev))
        f1score = 2*(precision*recall)/(precision+recall)
        roc_auc = roc_auc_score(y, probs)
        threshold_metrics.loc[t,['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC']] = [recall,precision,f1score,accuracy,specificity,ppv,npv,roc_auc]

    ### Find results for best threshold
    condition1 = threshold_metrics.loc[:,metric_of_interest] < desired_metric_value + error
    condition2 = threshold_metrics.loc[:,metric_of_interest] > desired_metric_value - error
    combined_condition = condition1 & condition2
    if metric_of_interest == 'Recall':
        sort_col = 'Precision'
    elif metric_of_interest == 'Precision':
        sort_col = 'Recall'
    elif metric_of_interest == 'F1-Score':
        sort_col = 'F1-Score'
    sorted_results = threshold_metrics[combined_condition].sort_values(by=sort_col,ascending=False)
    #print(sorted_results)
    if len(sorted_results) > 0:
        """ Only Record Value if Condition is Satisfied """
        thresh_adj_results_df.loc[['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC'],1] = sorted_results.iloc[0,:]   
        best_threshold = sorted_results.iloc[0,:].name
    else:
        print('No Threshold Found for Constraint!')
    return best_threshold, thresh_adj_results_df

def confusion_matrix(y_pred, y):
    true_pos = np.sum((y_pred == 1) & (y == 1))
    false_pos = np.sum((y_pred == 1) & (y == 0))
    true_neg = np.sum((y_pred == 0) & (y == 0))
    false_neg = np.sum((y_pred == 0) & (y == 1))
    return true_neg, false_neg, false_pos, true_pos

def get_test_results_from_threshold(yhat_test, y_test, thresh,n1,n2):
    y = pd.Series(y_test)
    unique, frequency = np.unique(y_test,return_counts = True)
    pred = np.where(yhat_test>thresh,1,0)
    pred = pd.Series(pred)
    true_neg, false_neg, false_pos, true_pos = confusion_matrix(pred, y)
    accuracy_test = accuracy_score(pred, y)
    #recall
    recall_test =  true_pos / (false_neg + true_pos)
    #precision
    precision_test = true_pos / (false_pos + true_pos)
    #specificity
    specificity_test = true_neg/(true_neg+false_pos)
    #for ppv and npv, set prevalance 
    #prev = 1/(match_number+1)
    prev = frequency[1]/len(y_test)
    #ppv
    ppv_test = (recall_test* (prev))/(recall_test * prev + (1-specificity_test) * (1-prev))
    if true_neg== 0 and false_neg==0:
        npv_test = 0
    else:
        npv_test = (specificity_test* (1-prev))/(specificity_test * (1-prev) + (1-recall_test) * (prev))
    f1score_test = 2*(precision_test*recall_test)/(precision_test+recall_test)
    roc_auc_test = roc_auc_score(y, yhat_test)

    return accuracy_test, recall_test, precision_test, specificity_test, f1score_test, ppv_test, npv_test, roc_auc_test


def calc_metrics(y, pred, pred_raw):
    probs = pred_raw
    y = pd.Series(y)
    print('num samples: '+ str(len(y)))
    pred = pd.Series(pred)
    n1 = len(y[y == 1])
    n2 = len(y[y == 0])
    unique, frequency = np.unique(y,return_counts = True)
    true_neg, false_neg, false_pos, true_pos = confusion_matrix(pred, y)
    accuracy_test = accuracy_score(pred, y)
    #recall
    recall_test =  true_pos / (false_neg + true_pos)
    #precision
    precision_test = true_pos / (false_pos + true_pos)
    #specificity
    specificity_test = true_neg/(true_neg+false_pos)
    #for ppv and npv, set prevalance 
    #prev = 0.05
    prev = frequency[1]/len(y)
    #ppv
    ppv_test = (recall_test* (prev))/(recall_test * prev + (1-specificity_test) * (1-prev))
    if true_neg== 0 and false_neg==0:
        npv_test = 0
    else:
        npv_test = (specificity_test* (1-prev))/(specificity_test * (1-prev) + (1-recall_test) * (prev))
    f1score_test = 2*(precision_test*recall_test)/(precision_test+recall_test)
    f_mean_test = math.sqrt(precision_test*recall_test)
    g_mean_test = math.sqrt(specificity_test*recall_test)
    if (len(set(pred))>1) & (len(set(y))>1):
        roc_auc_test = roc_auc_score(y, probs)
    else:
        roc_auc_test = np.nan
    
    return recall_test, specificity_test, recall_score, specificity_score, ppv_score, npv_score, roc_auc_test, conf_l, conf_u
