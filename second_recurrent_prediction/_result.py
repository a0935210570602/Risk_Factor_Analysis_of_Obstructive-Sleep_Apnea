import os
import pandas as pd
from sklearn import metrics

# TODO
def gen_result(self, pred_y, name, train_auc, is_train=True):
    true_y = self.train_Y if is_train else self.test_Y
    label = 'train' if is_train else 'test'
    accuracy = round(metrics.accuracy_score(true_y, pred_y), 2)
    precision = round(metrics.precision_score(true_y, pred_y), 2)
    recall = round(metrics.recall_score(true_y, pred_y), 2)
    f1_score_value = round(metrics.f1_score(true_y, pred_y), 2)
    auc_curve = round(train_auc, 2)

    df = pd.DataFrame({
        'Model name': name,
        'dataset': label,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1-score': f1_score_value,
        'auc': auc_curve
    }, index=[0])

    if is_train:
        if self.train_result_df.empty:
            self.train_result_df = df.copy()
        else:
            self.train_result_df = pd.concat([self.train_result_df, df], axis=0, ignore_index=True)
    else:
        if self.test_result_df.empty:
            self.test_result_df = df.copy()
        else:
            self.test_result_df = pd.concat([self.test_result_df, df], axis=0, ignore_index=True)

    # Save the results to a CSV file
    ## Create a folder to save training data if it does not exist.
    dir_path = os.path.join(self.PATH, label)
    file_path = os.path.join(self.PATH, label, '{label}.csv')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    result_df = self.train_result_df if is_train else self.test_result_df 
    result_df.to_csv(file_path)
