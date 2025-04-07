import pandas as pd
import os
from sklearn import model_selection, metrics

# TESTED
def load_data(self):
    # Split data to Training set & Testing set
    self.data_df = pd.concat([self.stroke_df, self.normal_df], axis=0)
    self.normal_train_df, self.normal_test_df = model_selection.train_test_split(self.normal_df, train_size=0.8)
    self.stroke_train_df, self.stroke_test_df = model_selection.train_test_split(self.stroke_df, train_size=0.8)

    train_df = pd.concat([self.stroke_train_df, self.normal_train_df], axis = 0)
    test_df = pd.concat([self.stroke_test_df, self.normal_test_df], axis = 0)
    train_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    # Select features of training & testing data
    self.train_X = train_df[self.SELECTED_FEATURE_LIST]
    self.train_Y = train_df[self.LABEL_NAME]

    self.test_X = test_df[self.SELECTED_FEATURE_LIST]
    self.test_Y = test_df[self.LABEL_NAME]

    self.data_X = self.data_df[self.SELECTED_FEATURE_LIST]
    self.data_Y = self.data_df[self.LABEL_NAME]

# TODO
# 10-fold
def cross_validation(self, model, name):
    global ten_fold_avg_std_df


    # Perform 10-fold cross-validation
    scores = model_selection.cross_validate(model, self.data_X, self.data_Y, cv=10,
                            scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'),
                            return_train_score=True)

    # Convert the scores to a DataFrame
    df_score = pd.DataFrame(scores)

    # Calculate mean and standard deviation of the scores
    avg_scores = df_score.mean()
    std_scores = df_score.std()

    # Save the scores of each fold to a separate CSV file (overwrite if exists)
    file_path = os.path.join(self.PATH, '10-fold', f'{name}_10-fold_scores.csv')
    df_score.to_csv(file_path, index=False)

    # Generate DataFrame for average scores and standard deviations
    avg_result_df = pd.DataFrame({
        'Model name': [name],
        'dataset': ['10-fold'],
        'accuracy_mean': [avg_scores['test_accuracy']],
        'accuracy_std': [std_scores['test_accuracy']],
        'precision_mean': [avg_scores['test_precision']],
        'precision_std': [std_scores['test_precision']],
        'recall_mean': [avg_scores['test_recall']],
        'recall_std': [std_scores['test_recall']],
        'f1-score_mean': [avg_scores['test_f1']],
        'f1-score_std': [std_scores['test_f1']],
        'auc_mean': [avg_scores['test_roc_auc']],
        'auc_std': [std_scores['test_roc_auc']]
    })

    dir_path = os.path.join(self.PATH, '10-fold')
    file_path = os.path.join(dir_path, 'all_models_10-fold-avg-std.csv')
    if not os.path.isdir(dir_path):  # 確認儲存檔案位置 若沒有的話 則新建檔案
        os.makedirs(dir_path)
    # Update the global DataFrame (remove any previous entry for the same model)
    if not self.ten_fold_avg_std_df.empty:
        print(self.ten_fold_avg_std_df.head(5))
        print(avg_result_df.head(5))
        self.ten_fold_avg_std_df = self.ten_fold_avg_std_df[self.ten_fold_avg_std_df['Model name'] != name]
        self.ten_fold_avg_std_df = pd.concat([self.ten_fold_avg_std_df, avg_result_df], axis=0, ignore_index=True)
    else:
        self.ten_fold_avg_std_df = avg_result_df.copy()

    # Save the average and standard deviation summary (overwrite if exists)
    self.ten_fold_avg_std_df.to_csv(file_path, index=False)

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

def display_total_result_train(self):
    print(self.train_result_df)

def display_total_result_test(self):
    print(self.test_result_df)

def display_total_ten_fold_result(self):
    print(self.ten_fold_avg_std_df)